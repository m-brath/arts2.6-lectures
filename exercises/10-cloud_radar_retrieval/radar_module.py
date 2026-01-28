#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Radar simulation and retrieval module for ARTS.
This module provides classes and functions for simulating cloud radar observations
and performing optimal estimation retrievals of hydrometeor properties using the
Atmospheric Radiative Transfer Simulator (ARTS).
Classes
ARTSConfig
    Base configuration class for ARTS simulations, managing paths, species, and
    basic settings for flux and radar simulations.
RadarSimulator
    Main class for radar forward simulations and retrievals. Inherits from ARTSConfig
    and provides methods for configuring scattering properties, defining hydrometeor
    species, running forward simulations, and performing optimal estimation retrievals.
Functions
---------
empirical_FWC_Z_relation(Z, a=0.137, b=0.643)
    Convert radar reflectivity to frozen water content using empirical Z-FWC relation.
empirical_RWC_Z_relation(Z, A=200, b=1.6, alpha=20.89, beta=1.15)
    Convert radar reflectivity to rain water content using empirical Z-R-RWC relation.
radar_reflectivity_to_apriori(y_cpr, z_cpr, z_background, sep_min_altitude, 
                               sep_max_altitude, hydrometeors=['liquid', 'frozen'],
                               min_val=1e-18, epsilon=1e-60)
    Generate a priori ice and rain water content profiles from radar reflectivity
    measurements with altitude-based phase separation.
generate_gridded_field_from_profiles(pressure_profile, temperature_profile, 
                                     z_field=None, gases={}, particulates={})
    Create a GriddedField4 object from 1D profiles of atmospheric variables.
add_scat_species(background_atm, species_name, type_name, 
                 scat_species_profile=[], pressure_profile=None)
    Add a scattering species profile to an existing background atmosphere object.
This module requires the pyarts package and associated data files for scattering
properties and spectroscopic line catalogs.
>>> # Initialize radar simulator
>>> radar = RadarSimulator()
>>> radar.set_frequency_grid([94e9])  # 94 GHz W-band radar
>>> # Define ice hydrometeor with Field et al. 2007 PSD
>>> radar.set_frozen_hydrometeor('IWC', 'mass_density', 
...                               PSD='FieldEtAl07TR',
...                               scatterer='H2O_ice_full_spectrum')
>>> # Run forward simulation
>>> result = radar.cloud_radar_1D(atm, min_range_bin_altitude=2000,
...                               max_range_bin_altitude=20000)
>>> reflectivity = result['Z']
>>> # Perform retrieval
>>> Hyd_ret, DeltaHyd, y_fit, result = radar.hydrometeor_retrieval(
...     y_obs=observations, S_y=obs_error_cov, S_a=prior_error_cov,
...     background_atm=atm, retrieval_quantities=['IWC-mass_density'])

Author
Manfred Brath
pyarts : Python interface to ARTS
radar_agenda : Module containing agenda definitions for radar simulations

"""
# %%
import os
import numpy as np
from pyarts import cat, xml, arts
from pyarts.workspace import Workspace, arts_agenda
import radar_agenda as ra

# %%


class ARTSConfig:
    """
    This class defines the basic setup for the flux simulator.
    """

    def __init__(self):
        """
        Parameters
        ----------
        setup_name : str
            Name of the setup. This name is used to create the directory for the LUT.

        Returns
        -------
        None.
        """

        cat.download.retrieve(verbose=True)

        # set default species
        self.species = [
            "H2O-PWR2022",
            "O2-PWR2022",
            "N2-SelfContPWR2021",
        ]

        # set some default values for some well mixed species
        self.well_mixed_species_defaults = {}
        self.well_mixed_species_defaults["O2"] = 0.21
        self.well_mixed_species_defaults["N2"] = 0.78

        # set default paths
        datapath = arts.globals.parameters.datapath

        self.basename_catalog = [str(x) for x in datapath if "arts-cat-data" in str(x)][
            0
        ]
        self.basename_scatterer = os.path.join(
            [str(x) for x in datapath if "arts-xml-data" in str(x)][0], "scattering"
        )
        self.scattering_data_from_arts_xml_package = True

        # set default value for polarization
        self.stokes_dim = 2

        # set if gas scattering is used
        self.gas_scattering = False

        # retrieval quantity list
        self.retrieval_quantities = []

    def set_paths(
        self,
        basename_scatterer=None,
    ):

        if basename_scatterer is not None:
            self.basename_scatterer = basename_scatterer

        else:
            datapath = arts.globals.parameters.datapath
            self.basename_scatterer = os.path.join(
                [str(x) for x in datapath if "arts-xml-data" in str(x)][0], "scattering"
            )

    def get_paths(self):
        """
        This function returns the paths as a dictionary.

        Returns
        -------
        Paths : dict
            Dictionary containing the paths.
        """

        Paths = {}
        Paths["basename_scatterer"] = self.basename_scatterer

        return Paths

    def print_paths(self):
        """
        This function prints the paths.

        Returns
        -------
        None.
        """
        print("basename_scatterer: ", self.basename_scatterer)


class RadarSimulator(ARTSConfig):

    def __init__(self):
        """
        This class defines the ARTS setup.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """

        super().__init__()

        # start ARTS workspace
        self.ws = Workspace()
        self.ws.verbositySetScreen(level=2)
        self.ws.verbositySetAgenda(level=0)

        # Set stoke dimension
        self.ws.IndexSet(self.ws.stokes_dim, self.stokes_dim)

        # Create my defined agendas in ws
        (
            self.ws,
            self.pnd_agenda_list,
        ) = ra.create_agendas_in_ws(self.ws)

        self.ws = ra.set_pnd_agendas_SB06(self.ws)
        self.ws = ra.set_pnd_agendas_MY05(self.ws)
        self.ws = ra.set_pnd_agendas_CG(self.ws)
        self.ws = ra.set_additional_pnd_agendas(self.ws)

        # Initialize scattering variables
        self.ws.ScatSpeciesInit()
        self.ws.ArrayOfArrayOfScatteringMetaDataCreate("scat_meta_temp")
        self.ws.ArrayOfArrayOfSingleScatteringDataCreate("scat_data_temp")

        # select/define agendas
        # =============================================================================

        self.ws.PlanetSet(option="Earth")

        self.ws.iy_space_agendaSet(option="CosmicBackground")
        self.ws.iy_cloudbox_agendaSet(option="LinInterpField")
        self.ws.water_p_eq_agendaSet()
        self.ws.ppath_step_agendaSet(option="GeometricPath")
        self.ws.ppath_agendaSet(option="FollowSensorLosPath")

        # define environment
        # =============================================================================

        # No jacobian calculations
        self.ws.jacobianOff()

        self.ws.cloudboxOff()

        # set absorption species
        self.ws.abs_speciesSet(species=self.species)

        self.ws.abs_lines_per_speciesSetEmpty()

    def set_frequency_grid(self, f_grid):
        """
        Set the frequency grid for the radar simulation.
        This method sets the frequency grid used in the radar calculations. The frequency
        grid must be provided as a list or array and all frequencies must be below 1 THz.
        Parameters
        ----------
        f_grid : list or numpy.ndarray
            Frequency grid in Hz. All values must be less than 1e12 Hz (1 THz).
        Raises
        ------
        ValueError
            If f_grid is not a list or numpy array.
        ValueError
            If any frequency in f_grid exceeds 1 THz (1e12 Hz).
        Examples
        --------
        >>> radar.set_frequency_grid([94e9, 95e9, 96e9])  # Set frequencies at 94, 95, 96 GHz
        >>> radar.set_frequency_grid(np.linspace(90e9, 100e9, 100))  # 100 frequencies from 90-100 GHz
        """

        # check that f_grid is a list or array
        if not isinstance(f_grid, (list, np.ndarray)):
            raise ValueError("f_grid must be a list or array.")

        # check that maximum frequency is less than 1 THz
        if max(f_grid) > 1e12:
            raise ValueError("Maximum frequency must be less than 1 THz.")

        # set frequency grid
        self.ws.f_grid = f_grid

    def set_species(self, species):
        """
        This function sets the gas absorption species.

        Parameters
        ----------
        species : list
            List of species.

        Returns
        -------
        None.
        """

        self.species = species
        self.ws.abs_species = self.species

    def get_species(self):
        """
        This function returns the gas absorption species.

        Returns
        -------
        list
            List of species.
        """

        return self.ws.abs_species

    def check_abs_species(self):
        """
        This function checks if all species are included in the atm_fields_compact
        that are defined in abs_species. If not, the species are added with the default
        values from well_mixed_species_defaults.
        A ValueError is raised if a species is not included in the atm_fields_compact and
        not in well_mixed_species_defaults.

        Returns
        -------
        None.
        """

        atm_grids = self.ws.atm_fields_compact.value.grids[0]

        # Get species of atm-field
        atm_species = [
            str(tag).split("-")[1] for tag in atm_grids if "abs_species" in str(tag)
        ]

        # Get species from defined abs_species
        abs_species = self.get_species().value
        abs_species = [str(tag).split("-")[0] for tag in abs_species]

        for abs_species_i in abs_species:

            if abs_species_i not in atm_species:

                # check for default
                if abs_species_i in self.well_mixed_species_defaults.keys():

                    self.ws.atm_fields_compactAddConstant(
                        self.ws.atm_fields_compact,
                        f"abs_species-{abs_species_i}",
                        self.well_mixed_species_defaults[abs_species_i],
                    )

                    print(
                        f"{abs_species_i} data not included in atmosphere data\n"
                        f"I will use default value {self.well_mixed_species_defaults[abs_species_i]}"
                    )

                else:

                    self.ws.atm_fields_compactAddConstant(
                        self.ws.atm_fields_compact,
                        f"abs_species-{abs_species_i}",
                        0.0,
                    )

                    print(
                        f"{abs_species_i} data not included in atmosphere data\n"
                        f"and it is not in well_mixed_species_defaults\n"
                        f"I will add this species with value 0."
                    )

    def define_particulate_scatterer(
        self,
        hydrometeor_type,
        pnd_agenda,
        scatterer_name,
        moments,
        scattering_data_folder=None,
    ):
        """
        Define a particulate scatterer species for radar simulations.

        This method configures a hydrometeor scatterer by setting up its particle number
        density (PND) agenda, loading scattering data, and adding it to the workspace's
        scattering species list.

        Parameters
        ----------
        hydrometeor_type : str
            Type of hydrometeor (e.g., 'ice', 'rain', 'snow'). Used as the species
            identifier string.
        pnd_agenda : str
            Name of the particle number density agenda to use for this scatterer.
            Should correspond to a workspace agenda (e.g., 'pnd_agenda_rain').
        scatterer_name : str
            Name identifier for the scatterer, used to locate scattering data files.
        moments : list of str
            List of moment names required by the PND agenda (e.g., ['mass_density',
            'number_density']). These will be combined with hydrometeor_type to form
            input names like '{hydrometeor_type}-{moment}'.
        scattering_data_folder : str, optional
            Path to the folder containing scattering data files. If None, defaults to
            self.basename_scatterer.

        Notes
        -----
        The method attempts to load scattering data in two ways:
        1. If scattering_data_from_arts_xml_package is True, it tries to load from
            MieSpheres XML files. On failure, falls back to SingleScatteringFile format.
        2. Otherwise, loads directly from XML files named after scatterer_name.

        The loaded scattering data and metadata are appended to the workspace's
        scat_data_raw and scat_meta arrays respectively.

        Raises
        ------
        RuntimeError
            If scattering data files cannot be found or loaded properly.
        """

        if scattering_data_folder is None:
            scattering_data_folder = self.basename_scatterer

        self.ws.StringCreate("species_id_string")
        self.ws.StringSet(self.ws.species_id_string, hydrometeor_type)
        self.ws.ArrayOfStringSet(
            self.ws.pnd_agenda_input_names,
            [f"{hydrometeor_type}-{moment}" for moment in moments],
        )

        self.ws.Append(self.ws.pnd_agenda_array, eval(f"self.ws.{pnd_agenda}"))
        self.ws.Append(self.ws.scat_species, self.ws.species_id_string)
        self.ws.Append(
            self.ws.pnd_agenda_array_input_names, self.ws.pnd_agenda_input_names
        )

        if self.scattering_data_from_arts_xml_package:
            try:
                ssd_name = os.path.join(
                    scattering_data_folder,
                    scatterer_name,
                    f"MieSpheres_{scatterer_name.replace('_full_spectrum','')}.xml",
                )
                self.ws.ReadXML(self.ws.scat_data_temp, ssd_name)
                smd_name = os.path.join(
                    scattering_data_folder,
                    scatterer_name,
                    f"MieSpheres_{scatterer_name.replace('_full_spectrum','')}.meta.xml",
                )
                self.ws.ReadXML(self.ws.scat_meta_temp, smd_name)
                self.ws.Append(self.ws.scat_data_raw, self.ws.scat_data_temp)
                self.ws.Append(self.ws.scat_meta, self.ws.scat_meta_temp)

            except RuntimeError:
                ssd_list = xml.load(
                    os.path.join(
                        self.basename_scatterer,
                        scatterer_name,
                        f"SingleScatteringFile_all{scatterer_name.replace('_','')}.xml",
                    )
                )
                self.ws.ScatSpeciesScatAndMetaRead(scat_data_files=ssd_list)

        else:
            ssd_name = os.path.join(scattering_data_folder, f"{scatterer_name}.xml")
            self.ws.ReadXML(self.ws.scat_data_temp, ssd_name)
            smd_name = os.path.join(
                scattering_data_folder, f"{scatterer_name}.meta.xml"
            )
            self.ws.ReadXML(self.ws.scat_meta_temp, smd_name)
            self.ws.Append(self.ws.scat_data_raw, self.ws.scat_data_temp)
            self.ws.Append(self.ws.scat_meta, self.ws.scat_meta_temp)

    def define_mie_graupel_scheme(
        self, RWC=True, LWC=True, IWC=True, SWC=True, GWC=True
    ):

        if RWC:
            self.define_particulate_scatterer(
                "RWC", "pnd_agenda_CGRWC", "H2O_liquid_full_spectrum", ["mass_density"]
            )
        if LWC:
            self.define_particulate_scatterer(
                "LWC", "pnd_agenda_CGLWC", "H2O_liquid_full_spectrum", ["mass_density"]
            )
        if IWC:
            self.define_particulate_scatterer(
                "IWC", "pnd_agenda_CGIWC", "H2O_ice_full_spectrum", ["mass_density"]
            )
        if SWC:
            self.define_particulate_scatterer(
                "SWC",
                "pnd_agenda_CGSWC_tropic",
                "H2O_ice_full_spectrum",
                ["mass_density"],
            )
        if GWC:
            self.define_particulate_scatterer(
                "GWC", "pnd_agenda_CGGWC", "H2O_ice_full_spectrum", ["mass_density"]
            )

    def define_mie_graupel_scheme_MW(self):

        self.define_particulate_scatterer(
            "RWC", "pnd_agenda_CGRWC", "H2O_liquid", ["mass_density"]
        )
        self.define_particulate_scatterer(
            "LWC", "pnd_agenda_CGLWC", "H2O_liquid", ["mass_density"]
        )
        self.define_particulate_scatterer(
            "IWC", "pnd_agenda_CGIWC", "H2O_ice", ["mass_density"]
        )
        self.define_particulate_scatterer(
            "SWC", "pnd_agenda_CGSWC_tropic", "H2O_ice", ["mass_density"]
        )
        self.define_particulate_scatterer(
            "GWC", "pnd_agenda_CGGWC", "H2O_ice", ["mass_density"]
        )

    def define_mie_MilbrandtYau_scheme(
        self, RWC=True, LWC=True, IWC=True, SWC=True, GWC=True, HWC=True
    ):

        if RWC:
            self.define_particulate_scatterer(
                "RWC",
                "pnd_agenda_MY05RWC",
                "LiquidSphere_Id25.scat_data",
                ["mass_density", "number_density"],
            )
        if LWC:
            self.define_particulate_scatterer(
                "LWC",
                "pnd_agenda_MY05LWC",
                "LiquidSphere_Id25.scat_data",
                ["mass_density", "number_density"],
            )
        if IWC:
            self.define_particulate_scatterer(
                "IWC",
                "pnd_agenda_MY05IWC",
                "GemCloudIce_Id31.scat_data",
                ["mass_density", "number_density"],
            )
        if SWC:
            self.define_particulate_scatterer(
                "SWC",
                "pnd_agenda_MY05SWC",
                "GemSnow_Id32.scat_data",
                ["mass_density", "number_density"],
            )
        if GWC:
            self.define_particulate_scatterer(
                "GWC",
                "pnd_agenda_MY05GWC",
                "GemGraupel_Id33.scat_data",
                ["mass_density", "number_density"],
            )
        if HWC:
            self.define_particulate_scatterer(
                "HWC",
                "pnd_agenda_MY05HWC",
                "GemHail_Id34.scat_data",
                ["mass_density", "number_density"],
            )

    def set_hydrometeor(self, hydrometeor, quantity, PSD, scatterer):
        """
        Set up hydrometeor scattering properties for radar simulation.
        This method configures the scattering characteristics of a hydrometeor species
        by setting the appropriate paths and defining the particulate scatterer with
        the specified particle size distribution (PSD) and scattering data.
        Parameters
        ----------
        hydrometeor : str
            Name of the hydrometeor species (e.g., 'cloud_ice', 'rain', 'snow').
        quantity : str or list
            Atmospheric quantity or quantities associated with the hydrometeor
            (e.g., 'IWC' for ice water content, 'RWC' for rain water content).
        PSD : str
            Particle size distribution type to be used (e.g., 'FieldEtAl07TR',
            ). This will be appended to 'pnd_agenda_'. see radar.pnd_agenda_list for available options.
        scatterer : str
            Name of the scattering data to use. Special cases include:
            - 'H2O_ice_full_spectrum': Uses built-in ARTS XML ice scattering data
            - 'H2O_liquid_full_spectrum': Uses built-in ARTS XML liquid scattering data
            - Other values: Uses custom scattering data from the 'scattering_data'
              directory in the current working directory.
        Notes
        -----
        The method automatically determines whether to use scattering data from the
        ARTS XML package or from a custom directory based on the scatterer parameter.
        For full spectrum water scatterers, the default ARTS paths are used. For
        custom scatterers, the method looks for data in a 'scattering_data'
        subdirectory of the current working directory.
        Examples
        --------
        >>> radar.set_hydrometeor('cloud_ice', 'IWC', 'FieldEtAl07TR',
        ...                       'H2O_ice_full_spectrum')
        >>> radar.set_hydrometeor('rain', 'RWC', 'AbelBoutle12', 'custom_rain_scatterer')
        """

        if (
            scatterer == "H2O_ice_full_spectrum"
            or scatterer == "H2O_liquid_full_spectrum"
        ):
            self.set_paths()
            self.scattering_data_from_arts_xml_package = True
        else:
            self.set_paths(os.path.join(os.getcwd(), "scattering_data"))
            self.scattering_data_from_arts_xml_package = False

        self.define_particulate_scatterer(
            hydrometeor,
            f"pnd_agenda_{PSD}",
            scatterer,
            [quantity],
        )

    def set_retrieval_quantity(self, hydrometeor, quantity, PSD, scatterer):
        """
        Set a retrieval quantity for a specific hydrometeor.
        This method configures a hydrometeor with its associated properties and adds it
        to the list of retrieval quantities in the format "{hydrometeor}-{quantity}".
        Parameters
        ----------
        hydrometeor : str
            The type of hydrometeor (e.g., 'rain', 'snow', 'ice', 'cloud_water').
        quantity : str
            The physical quantity to retrieve (e.g., 'content', 'mass', 'number_density').
        PSD : str
            Particle Size Distribution name, for example FieldEtAl07TR. See radar.pnd_agenda_list for available options.
        scatterer : str
            Scatterer name defining the scattering properties of the hydrometeor.
        Returns
        -------
        None
        Notes
        -----
        The retrieval quantity is stored in the format "{hydrometeor}-{quantity}" and
        appended to the `retrieval_quantities` list.
        Examples
        --------
        >>> radar.set_retrieval_quantity('RWC', 'mass_density', 'AbelBoutle12', 'custom_rain_scatterer')
        >>> print(radar.retrieval_quantities)
        ['RWC-mass_density']
        """

        self.set_hydrometeor(hydrometeor, quantity, PSD, scatterer)
        retrieval_quantity = f"{hydrometeor}-{quantity}"
        self.retrieval_quantities.append(retrieval_quantity)

    def set_liquid_hydrometeor(
        self,
        hydrometeor,
        quantity,
        PSD="AbelBoutle12",
        scatterer="H2O_liquid_full_spectrum",
    ):
        """
        Set a liquid hydrometeor species for the radar simulation and add it to retrieval quantities.
        This method configures a liquid hydrometeor (e.g., cloud droplets, rain) by setting its
        particle size distribution (PSD) and scattering properties, then registers the specified
        quantity as a retrieval parameter.
        Parameters
        ----------
        hydrometeor : str
            The name/type of the liquid hydrometeor species (e.g., 'cloud_water', 'rain').
        quantity : str
            The physical quantity to be retrieved (e.g., 'mass_density', 'number_density').
        PSD : str, optional
            The particle size distribution model to use. Default is 'AbelBoutle12'.
        scatterer : str, optional
            The scattering database identifier for liquid water.
            Default is 'H2O_liquid_full_spectrum'.
        Notes
        -----
        The method internally calls `set_hydrometeor()` to configure the hydrometeor
        properties and then appends the combination of hydrometeor and quantity
        (formatted as '{hydrometeor}-{quantity}') to the retrieval quantities list.
        Examples
        --------
        >>> radar.set_liquid_hydrometeor('cloud_water', 'mass_density')
        >>> radar.set_liquid_hydrometeor('SWC', 'mass_density', PSD='FieldEtAl07TR', scatterer='custom_frozen_scatterer')
        """

        self.set_hydrometeor(hydrometeor, quantity, PSD, scatterer)
        retrieval_quantity = f"{hydrometeor}-{quantity}"
        self.retrieval_quantities.append(retrieval_quantity)

    def set_frozen_hydrometeor(
        self,
        hydrometeor,
        quantity,
        PSD="FieldEtAl07TR",
        scatterer="H2O_ice_full_spectrum",
    ):
        """
        Set a frozen hydrometeor species with its properties for radar simulation.
        This method configures a frozen hydrometeor (ice-phase particle) for the radar
        simulation and marks it as a retrieval quantity. It wraps the `set_hydrometeor`
        method and automatically adds the hydrometeor-quantity pair to the list of
        retrieval quantities.
        Parameters
        ----------
        hydrometeor : str
            The type of frozen hydrometeor (e.g., 'SWC', 'snow', 'graupel').
        quantity : str
            The quantity to be set for the hydrometeor (e.g., 'mass_density').
        PSD : str, optional
            The particle size distribution model to use. Default is 'FieldEtAl07TR'.
        scatterer : str, optional
            The scattering properties model for ice particles. Default is
            'H2O_ice_full_spectrum'.
        Returns
        -------
        None
        Notes
        -----
        - This method is specifically designed for frozen (ice-phase) hydrometeors.
        - The retrieval quantity is automatically appended to `self.retrieval_quantities`
          in the format '{hydrometeor}-{quantity}'.
        - The actual hydrometeor configuration is delegated to the `set_hydrometeor`
          method.
        See Also
        --------
        set_hydrometeor : Base method for setting hydrometeor properties.
        """

        self.set_hydrometeor(hydrometeor, quantity, PSD, scatterer)
        retrieval_quantity = f"{hydrometeor}-{quantity}"
        self.retrieval_quantities.append(retrieval_quantity)

    def prepare_sensor(self, trans_pol="v", unit="Ze"):
        """
        Prepare the sensor for radar retrieval.
        Args:
            trans_pol (str, optional): Transmission polarization. Defaults to "v".
            unit (str, optional): Unit for radar retrieval. Defaults to "Ze".
        Raises:
            ValueError: If trans_pol is not "v" or "h".
        Returns:
            None
        """

        if trans_pol == "v":
            self.ws.ArrayOfIndexSet(self.ws.instrument_pol, [1])
        elif trans_pol == "h":
            self.ws.ArrayOfIndexSet(self.ws.instrument_pol, [1])
        else:
            raise ValueError(
                "only vertical (v) and horizontal (h) polarization are supported."
            )

        self.ws.iy_radar_agenda = ra.iy_radar_agenda_singlepol(self.ws)
        self.ws.IndexSet(self.ws.stokes_dim, 1)
        self.ws.sensorOff()
        self.ws.StringSet(self.ws.iy_unit_radar, unit)

    def set_sensor_position_and_view(self, sensor_pos, sensor_los):

        self.ws.sensor_pos = np.array([[sensor_pos]])
        self.ws.sensor_los = np.array([[sensor_los]])

    def basic_settings(
        self,
        atm,
        min_range_bin_altitude=2000.0,
        max_range_bin_altitude=20000.0,
        sensor_altitude=100000.0,
        sensor_los=180.0,
        N_range_bins_edges=91,
    ):
        """
        Configure basic radar simulation settings including atmosphere, sensor position, and range bins.
        This method sets up the fundamental parameters required for radar simulation, including
        atmospheric conditions, sensor positioning, scattering properties, and range bin configuration.
        Parameters
        ----------
        atm : AtmFieldsCompact
            Compact atmospheric fields containing temperature, pressure, and species concentrations.
        min_range_bin_altitude : float, optional
            Minimum altitude for range bins in meters. Default is 2000.0 m.
        max_range_bin_altitude : float, optional
            Maximum altitude for range bins in meters. Default is 20000.0 m.
        sensor_altitude : float, optional
            Altitude of the sensor platform in meters. Default is 100000.0 m.
        sensor_los : float, optional
            Sensor line-of-sight angle in degrees. Default is 180.0 (nadir).
        N_range_bins_edges : int, optional
            Number of range bin edges (creates N-1 range bins). Default is 91.
        Notes
        -----
        - Sets atmosphere dimension to 1D
        - Automatically configures propagation matrix for clear sky conditions
        - Sets surface altitude to match the lowest atmospheric level
        - Calculates and validates scattering data with interpolation order 1
        - Configures instrument polarization array for all frequency grid points
        - Range bins are evenly spaced between min and max altitudes
        - The `range_bins` attribute stores bin centers (excluding the last edge)
        Side Effects
        ------------
        - Modifies multiple workspace (ws) attributes
        - Sets `self.range_bins` to contain range bin centers
        """

        # prepare atmosphere
        self.ws.atmosphere_dim = 1
        self.ws.atm_fields_compact = atm
        self.check_abs_species()
        self.ws.AtmFieldsAndParticleBulkPropFieldFromCompact()

        self.ws.propmat_clearsky_agendaAuto()

        # surface altitudes
        self.ws.z_surface = np.ones((1, 1)) * self.ws.z_field.value[0]

        # prepare scattering data
        self.ws.scat_dataCalc(interp_order=1)
        self.ws.scat_dataCheck(check_type="sane")

        self.ws.instrument_pol_array = []
        for i in range(len(self.ws.f_grid.value)):
            self.ws.Append(self.ws.instrument_pol_array, self.ws.instrument_pol)

        # set sensor position
        self.set_sensor_position_and_view(sensor_altitude, sensor_los)

        # set up range bins
        range_bins = np.linspace(
            min_range_bin_altitude, max_range_bin_altitude, N_range_bins_edges
        )
        self.ws.range_bins = range_bins
        self.range_bins = range_bins[0:-1]

    def set_retrieval_quantities(
        self,
        retrieval_species=[],
        retrieval_quantities=[],
    ):
        if len(retrieval_quantities) > 0:
            self.ws.jacobianInit()

            for i in range(len(retrieval_quantities)):
                self.ws.jacobianAddScatSpecies(
                    g1=self.ws.p_grid,
                    g2=self.ws.lat_grid,
                    g3=self.ws.lon_grid,
                    species=retrieval_species[i],
                    quantity=retrieval_quantities[i],
                )
            self.ws.jacobianClose()

    def do_checks(self):

        self.ws.sensor_checkedCalc()
        self.ws.atmfields_checkedCalc()
        self.ws.atmgeom_checkedCalc()
        self.ws.cloudbox_checkedCalc()

        # Here we set the check level to none, because we already checked the data
        self.ws.scat_data_checkedCalc(check_level="none")

    def prepare_output(self, retrieval_quantities):

        dbZ_data = self.ws.y.value[:] * 1.0
        dbZ = np.reshape(dbZ_data, (len(self.ws.f_grid.value), len(self.range_bins)))

        data = {
            "Z": dbZ,
            "Z_raw": dbZ_data,
            "range_bins": self.range_bins,
            "frequencies": self.ws.f_grid.value,
            "range_bin_units": "m",
            "Z_units": "dBZe",
            "frequency_units": "Hz",
        }

        if len(retrieval_quantities) > 0:
            jacobian = self.ws.jacobian.value[:].copy()
            shape = jacobian.shape
            cols = shape[1] // len(retrieval_quantities)

            for i, rq in enumerate(retrieval_quantities):
                jacobian_rq = jacobian[:, i * cols : (i + 1) * cols]
                data[f"jacobian_{rq}"] = np.reshape(
                    jacobian_rq,
                    (
                        len(self.ws.f_grid.value),
                        len(self.range_bins),
                        jacobian_rq.shape[1],
                    ),
                )

        return data

    def cloud_radar_1D(
        self,
        atm,
        min_range_bin_altitude=2000.0,
        max_range_bin_altitude=20000.0,
        sensor_altitude=100000.0,
        sensor_los=180.0,
        N_range_bins_edges=91,
        dbze_min=-35.0,
        retrieval_quantities=[],
    ):

        self.basic_settings(
            atm=atm,
            min_range_bin_altitude=min_range_bin_altitude,
            max_range_bin_altitude=max_range_bin_altitude,
            sensor_altitude=sensor_altitude,
            sensor_los=sensor_los,
            N_range_bins_edges=N_range_bins_edges,
        )

        retrieval_species = []
        for rq in retrieval_quantities:
            retrieval_species.append(rq.split("-")[0])

        self.set_retrieval_quantities(
            retrieval_species=retrieval_species,
            retrieval_quantities=retrieval_quantities,
        )

        self.ws.cloudboxSetFullAtm()
        self.ws.pnd_fieldCalcFromParticleBulkProps()

        self.do_checks()

        self.ws.yRadar(dbze_min=dbze_min)

        # prepare data output
        result = self.prepare_output(retrieval_quantities)

        return result

    def hydrometeor_retrieval(
        self,
        y_obs,
        S_y,
        S_a,
        background_atm,
        retrieval_quantities=[],
        min_range_bin_altitude=2000.0,
        max_range_bin_altitude=20000.0,
        sensor_altitude=100000.0,
        sensor_los=180.0,
        N_range_bins_edges=91,
        Verbosity=False,
        **kwargs,
    ):
        """
        Perform hydrometeor retrieval from radar observations.
        This method retrieves hydrometeor properties (e.g., ice water content, rain water content)
        from radar observations using optimal estimation methods.
        Parameters
        ----------
        y_obs : array_like
            Observation vector containing the measured radar reflectivity values.
        S_y : array_like
            Observation error covariance matrix representing measurement uncertainties.
        S_a : array_like
            A priori error covariance matrix representing background state uncertainties.
        background_atm : object
            Background atmospheric state used as the a priori in the retrieval.
        retrieval_quantities : list, optional
            List of hydrometeor quantities to retrieve (e.g., ['IWC', 'RWC']).
            Default is an empty list.
        min_range_bin_altitude : float, optional
            Minimum altitude of the range bins in meters. Default is 2000.0 m.
        max_range_bin_altitude : float, optional
            Maximum altitude of the range bins in meters. Default is 20000.0 m.
        sensor_altitude : float, optional
            Altitude of the sensor platform in meters. Default is 100000.0 m.
        sensor_los : float, optional
            Sensor line-of-sight angle in degrees. Default is 180.0° (nadir).
        N_range_bins_edges : int, optional
            Number of range bin edges for discretizing the vertical profile.
            Default is 91.
        Verbosity : bool, optional
            If True, enables verbose output during retrieval. Default is False.
        **kwargs : dict
            Additional keyword arguments passed to the underlying retrieval function.
        Returns
        -------
        Hyd_ret : dict
            Dictionary containing retrieved hydrometeor values for each quantity.
            Keys are the retrieval quantity names, values are 1D arrays of retrieved values.
        DeltaHyd : dict
            Dictionary containing total retrieval uncertainties for each quantity.
            Keys are the retrieval quantity names, values are 1D arrays of uncertainties
            computed from observation and smoothing errors.
        y_fit : array_like
            Fitted observation vector (simulated measurements based on retrieved state).
        result : dict
            Complete result dictionary from the retrieval containing all diagnostic
            information including state vector, Jacobian, averaging kernel, etc.
        Raises
        ------
        ValueError
            If retrieval_quantities is empty (no retrieval quantities defined).
        Notes
        -----
        The retrieval uses optimal estimation theory to combine observations with
        a priori information. The total uncertainty combines observation error (dx_o)
        and smoothing error (dx_s) in quadrature.
        """

        if len(retrieval_quantities) < 1:
            retrieval_quantities=self.retrieval_quantities

        self.basic_settings(
            atm=background_atm,
            min_range_bin_altitude=min_range_bin_altitude,
            max_range_bin_altitude=max_range_bin_altitude,
            sensor_altitude=sensor_altitude,
            sensor_los=sensor_los,
            N_range_bins_edges=N_range_bins_edges,
        )

        result = self.retrieval_hyd(
            y_obs, S_y, S_a, retrieval_quantities, Verbosity=Verbosity, **kwargs
        )

        Hyd_ret_temp = result["x"].reshape((len(retrieval_quantities), -1))
        DeltaHyd_temp = np.sqrt(result["dx_o"] ** 2 + result["dx_s"] ** 2)
        DeltaHyd_temp = DeltaHyd_temp.reshape((len(retrieval_quantities), -1))
        y_fit = result["y_fit"]

        Hyd_ret = {
            retrieval_quantities[i]: Hyd_ret_temp[i, :]
            for i in range(len(retrieval_quantities))
        }
        DeltaHyd = {
            retrieval_quantities[i]: DeltaHyd_temp[i, :]
            for i in range(len(retrieval_quantities))
        }

        return Hyd_ret, DeltaHyd, y_fit, result

    def retrieval_hyd(
        self,
        y,
        S_y,
        S_a,
        retrieval_quantities,
        max_iter=50,
        stop_dx=0.01,
        Verbosity=False,
        lm_ga_settings=[1e1, 2, 3, 1e9, 1, 99],
    ):
        """
        Perform optimal estimation method (OEM) retrieval for hydrometeor properties.
        This method performs an atmospheric retrieval using the optimal estimation method
        to retrieve scattering species (hydrometeors) from radar measurements. It sets up
        the retrieval problem, defines retrieval quantities, and runs the OEM iteration
        to find the optimal atmospheric state.
        Parameters
        ----------
        y : array_like
            Measurement vector containing the observed radar reflectivities.
        S_y : array_like
            Measurement error covariance matrix. Describes the uncertainty in the
            observations.
        S_a : array_like or list of array_like
            A priori error covariance matrix (or list of matrices if multiple retrieval
            quantities are defined). Describes the uncertainty in the a priori state.
        retrieval_quantities : list of str
            List of retrieval quantity strings. Each string should specify the species
            and quantity to retrieve (e.g., "species-quantity" format).
        max_iter : int, optional
            Maximum number of OEM iterations. Default is 50.
        stop_dx : float, optional
            Convergence criterion for the state vector change. The iteration stops when
            the change in state vector is smaller than this value. Default is 0.01.
        Verbosity : bool, optional
            If True, print detailed progress information during the retrieval.
            Default is False.
        lm_ga_settings : list, optional
            Levenberg-Marquardt gamma settings for the OEM solver. The list contains:
            [start_gamma, gamma_increase_factor, gamma_decrease_factor,
                max_gamma, lambda_threshold, max_lambda_iterations].
            Default is [1e1, 2, 3, 1e9, 1, 99].
        Returns
        -------
        dict
            Dictionary containing retrieval results with the following keys:
            - 'x' : Retrieved state vector
            - 'x_apr' : A priori state vector
            - 'y_fit' : Fitted measurement vector
            - 'S_o' : Observation error covariance matrix
            - 'S_s' : Smoothing error covariance matrix
            - 'dx_o' : Observation error (diagonal of S_o)
            - 'dx_s' : Smoothing error (diagonal of S_s)
            - 'A' : Averaging kernel matrix
            - 'G' : Gain matrix (contribution function)
        Raises
        ------
        ValueError
            If the length of S_a list does not match the number of retrieval quantities,
            or if S_a is not a list when multiple retrieval quantities are specified.
        Warning
            If a retrieval quantity cannot be added to the workspace.
        Notes
        -----
        - The method uses log10 transformation for the Jacobian.
        - NaN values in the particle number density (pnd) field are set to zero.
        - The retrieval uses the Levenberg-Marquardt optimization method.
        - Convergence diagnostics are printed if the retrieval converges successfully.
        Examples
        --------
        >>> result = radar.retrieval_hyd(
        ...     y=measurements,
        ...     S_y=measurement_error_cov,
        ...     S_a=prior_error_cov,
        ...     retrieval_quantities=["IWC-mass_density"],
        ...     max_iter=30,
        ...     stop_dx=0.01,
        ...     Verbosity=True
        ... )
        >>> retrieved_state = result['x']
        >>> averaging_kernel = result['A']
        """

        # Copy the measeurement vector to the ARTS workspace
        self.ws.y = y

        self.ws.cloudboxSetFullAtm()

        # Start definition of retrieval quantities
        ###########################################################################
        self.ws.retrievalDefInit()

        # check if S_a is a list
        if isinstance(S_a, list):
            if len(S_a) != len(retrieval_quantities):
                raise ValueError(
                    "Length of S_a must be equal to length of retrieval_quantities."
                )
        else:
            if len(retrieval_quantities) > 1:
                raise ValueError(
                    "If multiple retrieval quantities are defined, S_a must be a list of covariance matrices."
                )

            # make S_a a list
            S_a = [S_a]

        for i, rq in enumerate(retrieval_quantities):
            try:

                self.ws.retrievalAddScatSpecies(
                    g1=self.ws.p_grid,
                    g2=self.ws.lat_grid,
                    g3=self.ws.lon_grid,
                    species=rq.split("-")[0],
                    quantity=rq,
                )

                # idx=[ii for ii, xx in enumerate(ws.abs_species.value) if "H2O" in str(xx)][0]
                # species_string=str(ws.abs_species.value[idx])
                # ws.retrievalAddAbsSpecies(g1=ws.p_grid, g2=ws.lat_grid, g3=ws.lon_grid, species=species_string, unit="vmr")
            except RuntimeError:
                raise Warning(
                    "Could not add retrieval quantity " + rq + ". Skipping it. ",
                    "Please check if the species and quantity are correct.",
                )
                continue

            self.ws.jacobianSetFuncTransformation(transformation_func="log10")

            # Set a priori covariance matrix
            self.ws.covmat_sxAddBlock(block=S_a[i])

        # Set measurement error covariance matrix
        self.ws.covmat_seAddBlock(block=S_y)

        # Close retrieval definition
        self.ws.retrievalDefClose()
        ############################################################################

        self.ws.pnd_fieldCalcFromParticleBulkProps()

        # Initialise
        # x, jacobian and yf must be initialised
        self.ws.VectorSet(self.ws.x, [])
        self.ws.VectorSet(self.ws.yf, [])
        self.ws.MatrixSet(self.ws.jacobian, [])

        # Iteration agenda
        @arts_agenda
        def inversion_iterate_agenda(ws):

            ws.Ignore(ws.inversion_iteration_counter)

            # Map x to ARTS' variables
            ws.x2artsAtmAndSurf()

            # calculate pnd fields
            ws.pnd_fieldCalcFromParticleBulkProps()

            # To be safe, rerun some checks
            ws.atmfields_checkedCalc()
            ws.atmgeom_checkedCalc()

            ws.yRadar(y=ws.yf)
            # ws.Copy(ws.yf,ws.y)
            ws.jacobianAdjustAndTransform()

        #
        self.ws.inversion_iterate_agenda = inversion_iterate_agenda

        # check for nan in pnds
        pnd_field = self.ws.pnd_field.value[:, :, :, :]
        pnd_field[np.isnan(pnd_field)] = 0
        self.ws.pnd_field = pnd_field

        # some basic checks
        self.ws.atmfields_checkedCalc()
        self.ws.atmgeom_checkedCalc()
        self.ws.sensor_checkedCalc()
        self.ws.scat_data_checkedCalc()
        self.ws.cloudbox_checkedCalc()

        # create a priori
        self.ws.xaStandard()

        # Run OEM
        self.ws.OEM(
            method="lm",
            max_iter=max_iter,
            display_progress=int(Verbosity),
            stop_dx=stop_dx,
            lm_ga_settings=lm_ga_settings,
        )
        #
        if Verbosity == True:
            self.ws.Print(self.ws.oem_errors, 0)

        oem_diagostics = self.ws.oem_diagnostics.value[:]
        if oem_diagostics[0] > 0:
            print(f"Convergence status:                    {oem_diagostics[0]}")
            print(f"Start value of cost function:          {oem_diagostics[1]}")
            print(f"End value of cost function:            {oem_diagostics[2]}")
            print(f"End value of y-part of cost function:  {oem_diagostics[3]}")
            print(f"Number of iterations:                  {oem_diagostics[4]}\n")

        # Compute averaging kernel matrix
        self.ws.avkCalc()

        # Compute smoothing error covariance matrix
        self.ws.covmat_ssCalc()

        # Compute observation system error covariance matrix
        self.ws.covmat_soCalc()

        # Extract observation errors
        self.ws.retrievalErrorsExtract()

        result = {}
        result["x"] = self.ws.x.value[:] * 1.0
        result["x_apr"] = self.ws.xa.value[:] * 1.0
        result["y_fit"] = self.ws.yf.value[:] * 1.0
        result["S_o"] = self.ws.covmat_so.value[:] * 1.0
        result["S_s"] = self.ws.covmat_ss.value[:] * 1.0
        result["dx_o"] = self.ws.retrieval_eo.value[:] * 1.0
        result["dx_s"] = self.ws.retrieval_ss.value[:] * 1.0
        result["A"] = self.ws.avk.value[:] * 1.0
        result["G"] = self.ws.dxdy.value[:] * 1.0

        return result


# %% addional functions
def empirical_FWC_Z_relation(Z, a=0.137, b=0.643):
    """
    This function calculates the frozen water content from the radar reflectivity using an empirical relation.

    Default values are from
    Liu, C., and A. J. Illingworth, 2000:
    Toward More Accurate Retrievals of Ice Water Content from Radar Measurements of Clouds.
    J. Appl. Meteor. Climatol., 39, 1130–1146,
    https://doi.org/10.1175/1520-0450(2000)039<1130:TMAROI>2.0.CO;2.

    for a frequency of 94 GHz.

    Parameters
    ----------
    Z : array
        Radar reflectivity in dBZe.
    a : float
        Coefficient a in the empirical relation.
    b : float
        Exponent b in the empirical relation.

    Returns
    -------
    IWC : array
        IWC in kg/m^3.
    """

    Z_linear = 10 ** (Z / 10)  # convert dBZe to linear scale
    IWC = 0.137 * (Z_linear) ** 0.643 / 1e3

    return IWC


def empirical_RWC_Z_relation(Z, A=200, b=1.6, alpha=20.89, beta=1.15):
    """
    Convert radar reflectivity to rain water content using an empirical Z-R-RWC relation.
    This function applies the empirical relationship between radar reflectivity (Z),
    rain rate (R), and rain water content (RWC) based on the power-law relationships:
        Z = A * R^b
        R = alpha * RWC^beta
    Combining these gives: RWC = RWC_0 * (R_0/alpha)^(1/beta) * (Z/A)^(1/(b*beta))

    The conversion from rain rate to rain water content is according to:
    Geer, A. J. / Lopez, Philippe / Bauer, Peter
    Lessons Learnt from the 1D+ 4D-Var Assimilation of Rain and Cloud Affected Ssm/i Observations at Ecmwf
    2007

    The Z-R relation parameters are according to:
    Marshall, J. S., W. Hitschfeld, and K. L. S. Gunn, 1955: Advances in radar weather.
    Advances in Geophysics, Vol. 2, Academic Press, 1–56, https://doi.org/10.1016/S0065-2687(08)60310-6.

    Parameters
    ----------
    Z : float or array-like
        Radar reflectivity in dBZe (decibel relative to Z_e).
    A : float, optional
        Coefficient in the Z-R relation Z = A * R^b. Default is 200.
    b : float, optional
        Exponent in the Z-R relation Z = A * R^b. Default is 1.6.
    alpha : float, optional
        Coefficient in the R-RWC relation R = alpha * RWC^beta. Default is 20.89.
    beta : float, optional
        Exponent in the R-RWC relation R = alpha * RWC^beta. Default is 1.15.
    Returns
    -------
    float or array-like
        Rain water content in g/m^3.
    Notes
    -----
    - The function converts dBZe to linear scale before applying the relationship.
    - RWC_0 is set to 1.0 kg/m^3 and R_0 is set to 1.0 mm/h as reference values.
    - The output is converted from kg/m^3 to g/m^3 by dividing by 1e3.
    """

    Z_linear = 10 ** (Z / 10)  # convert dBZe to linear scale

    RWC_0 = 1.0  # kg/m^3
    R_0 = 1.0  # mm / h

    return (
        RWC_0 / 1e3 * (R_0 / alpha) ** (1 / beta) * (Z_linear / A) ** (1 / (b * beta))
    )


def radar_reflectivity_to_apriori(
    y_cpr,
    z_cpr,
    z_background,
    sep_min_altitude,
    sep_max_altitude,
    hydrometeors=["liquid", "frozen"],
    min_val=1e-18,
    epsilon=1e-60,
):
    """
    Convert radar reflectivity measurements to a priori ice and rain water content profiles.
    This function takes radar reflectivity data and converts it to ice water content (IWC)
    and rain water content (RWC) estimates on a background altitude grid. It applies a
    weighting function to smoothly transition between frozen and liquid hydrometeor regimes
    based on specified altitude boundaries.
    Parameters
    ----------
    y_cpr : array_like
        Radar reflectivity measurements in dBZ, shape (n_altitudes_cpr, n_profiles).
    z_cpr : array_like
        Altitude grid corresponding to the radar measurements, shape (n_altitudes_cpr,).
    z_background : array_like
        Background altitude grid for output profiles, shape (n_altitudes_background,).
    sep_min_altitude : float
        Minimum altitude (lower boundary) for the transition zone between liquid and frozen
        hydrometeors. Below this altitude, only liquid hydrometeors are considered.
    sep_max_altitude : float
        Maximum altitude (upper boundary) for the transition zone. Above this altitude, only
        frozen hydrometeors are considered.
    hydrometeors : list of str, optional
        List of hydrometeor types to include. Can contain "liquid" and/or "frozen".
        Default is ["liquid", "frozen"].
    min_val : float, optional
        Minimum value for water content in kg/m³. Used as a lower bound for interpolation
        and for altitudes outside the valid range. Default is 1e-18.
    epsilon : float, optional
        Small constant added to prevent log of zero in weighting
        calculations. Default is 1e-60.
    Returns
    -------
    IWC_apr : ndarray
        A priori ice water content profiles, shape (n_altitudes_background, n_profiles).
        Values in kg/m³.
    RWC_apr : ndarray
        A priori rain water content profiles, shape (n_altitudes_background, n_profiles).
        Values in kg/m³.
    Notes
    -----
    - The weighting function w varies linearly from 0 to 1 between sep_min_altitude and
      sep_max_altitude, controlling the transition between liquid and frozen phases.
    - When sep_max_altitude equals sep_min_altitude, a step function is used instead.
    - Empirical Z-W relationships are applied via `empirical_FWC_Z_relation` and
      `empirical_RWC_Z_relation` functions.
    - In the transition zone, reflectivity is weighted before applying empirical relations
      to account for mixed-phase conditions.
    """
    

    IWC_apr = np.zeros((len(z_background), np.size(y_cpr, 1)))
    RWC_apr = np.zeros((len(z_background), np.size(y_cpr, 1)))

    for idx in range(np.size(y_cpr, 1)):
        
        # calculate weighting function
        if sep_max_altitude==sep_min_altitude:
            w=np.ones_like(z_cpr)
            w[z_cpr<sep_min_altitude]=0
        else:
            w=(z_cpr-sep_min_altitude)/(sep_max_altitude-sep_min_altitude)
            w[z_cpr<sep_min_altitude]=0
            w[z_cpr>sep_max_altitude]=1
            
        
        if "frozen" in hydrometeors:
            IWC_Z = empirical_FWC_Z_relation(y_cpr[:, idx])
            IWC_apr[:, idx] = np.interp(
                z_background, z_cpr, IWC_Z, left=min_val, right=min_val
            )

            logic = z_background < sep_min_altitude
            IWC_apr[logic, idx] = min_val

            logic = np.logical_and(
                sep_min_altitude < z_background, z_background < sep_max_altitude
            )            
            dumb = empirical_FWC_Z_relation(y_cpr[:, idx] + 10 * np.log10(w+epsilon))
            dumb = np.interp(z_background, z_cpr, dumb, left=min_val, right=min_val)
            IWC_apr[logic, idx] = dumb[logic]

        if "liquid" in hydrometeors:
            RWC_Z = empirical_RWC_Z_relation(y_cpr[:, idx])
            RWC_apr[:, idx] = np.interp(
                z_background, z_cpr, RWC_Z, left=min_val, right=min_val
            )
            logic = z_background > sep_max_altitude
            RWC_apr[logic, idx] = min_val

            logic = np.logical_and(
                sep_min_altitude < z_background, z_background < sep_max_altitude
            )
            dumb = empirical_RWC_Z_relation(y_cpr[:, idx] + 10 * np.log10((1-w)+epsilon))
            dumb = np.interp(z_background, z_cpr, dumb, left=min_val, right=min_val)
            RWC_apr[logic, idx] = dumb[logic]                

    return IWC_apr, RWC_apr


def generate_gridded_field_from_profiles(
    pressure_profile, temperature_profile, z_field=None, gases={}, particulates={}
):
    """
    Generate a gridded field from profiles of pressure, temperature, altitude, gases and particulates.

    Parameters:
    -----------
    pressure_profile : array
        Pressure profile in Pa.

    temperature_profile : array
        Temperature profile in K.

    z_field : array, optional
        Altitude profile in m. If not provided, it is calculated from the pressure profile.

    gases : dict
        Dictionary with the gas species as keys and the volume mixing ratios as values.

    particulates : dict
        Dictionary with the particulate species with the name of quantity as keys and the quantity values.
        E.g. {'LWC-mass_density': LWC_profile} mass density of liquid water content in kg/m^3.
    Returns:
    --------
    atm_field : GriddedField4
        Gridded field with the profiles of pressure, temperature, altitude, gases and particulates.

    """

    atm_field = arts.GriddedField4()

    # Do some checks
    if len(pressure_profile) != len(temperature_profile):
        raise ValueError("Pressure and temperature profile must have the same length")

    if z_field is not None and len(pressure_profile) != len(z_field):
        raise ValueError("Pressure and altitude profile must have the same length")

    # Generate altitude field if not provided
    if z_field is None:
        z_field = 16e3 * (5 - np.log10(pressure_profile))

    # set up grids
    abs_species = [f"abs_species-{key}" for key in list(gases.keys())]
    scat_species = [f"scat_species-{key}" for key in list(particulates.keys())]
    atm_field.set_grid(0, ["T", "z"] + abs_species + scat_species)
    atm_field.set_grid(1, pressure_profile)

    # set up data
    atm_field.data = np.zeros((len(atm_field.grids[0]), len(atm_field.grids[1]), 1, 1))

    # The first two values are temperature and altitude
    atm_field.data[0, :, 0, 0] = temperature_profile
    atm_field.data[1, :, 0, 0] = z_field

    # The next values are the gas species
    for i, key in enumerate(list(gases.keys())):
        atm_field.data[i + 2, :, 0, 0] = gases[key]

    # The next values are the particulates
    for i, key in enumerate(list(particulates.keys())):
        atm_field.data[i + 2 + len(gases.keys()), :, 0, 0] = particulates[key]

    return atm_field


def add_scat_species(
    background_atm,
    species_name,
    type_name,
    scat_species_profile=[],
    pressure_profile=None,
):
    """
    Add a scattering species to the background atmosphere.
    This function adds a new scattering species profile to an existing background
    atmosphere object by appending it to the grid variables and data arrays.
    Parameters
    ----------
    background_atm : object
        Background atmosphere object containing grids and data attributes.
        - grids[0]: List of variable names
        - grids[1]: Pressure grid
        - grids[2], grids[3]: Latitude and Longitude grids (if applicable)
        - data: pyarts.arts.GriddedField4 or similar containing atmospheric data
    species_name : str
        Name of the scattering species (e.g., SWC', 'LWC', 'snow').
    type_name : str
        Type identifier for the scattering species (e.g., 'mass_density', 'number_density').
    scat_species_profile : array-like, int, or float, optional
        Scattering species concentration profile. Can be:
        - A scalar value to apply uniformly across all pressure levels
        - An array matching the background atmosphere pressure grid
        - An array with custom pressure levels (requires pressure_profile)
        Default is an empty list (zeros).
    pressure_profile : array-like or None, optional
        Custom pressure grid corresponding to scat_species_profile. If provided,
        the profile will be interpolated to match the background atmosphere's
        pressure grid. Default is None.
    Returns
    -------
    background_atm : object
        Modified background atmosphere object with the added scattering species.
    Raises
    ------
    ValueError
        If scat_species_profile size does not match the background atmosphere
        profile size when pressure_profile is None.
    Notes
    -----
    - The scattering species is added as a new variable with the naming convention:
      'scat_species-{species_name}-{type_name}'
    - If no profile is provided, the species concentration is set to zero at all levels
    - The function modifies the input background_atm object in place and returns it
    Examples
    --------
    >>> # Add uniform ice cloud
    >>> atm = add_scat_species(atm, 'ice', 'mass_density', 1e-6)
    >>>
    >>> # Add custom profile with interpolation
    >>> pressures = [100000, 50000, 10000]
    >>> profile = [0, 1e-5, 0]
    >>> atm = add_scat_species(atm, 'liquid', 'mass_density', profile, pressures)
    """

    variables = background_atm.grids[0]
    variables.append(f"scat_species-{species_name}-{type_name}")
    background_atm.set_grid(0, variables)
    data = background_atm.data.value
    data2 = np.concatenate((data, np.zeros((1, np.size(data, 1), 1, 1))), axis=0)

    if isinstance(scat_species_profile, (int, float)):
        data2[-1, :, 0, 0] = scat_species_profile
    elif len(scat_species_profile) > 0:
        if pressure_profile is None:
            # check that pressure profile
            if np.size(scat_species_profile) != background_atm.grids[1]:
                raise ValueError(
                    "Scat species profile size does not match background atmosphere profile size"
                )

        else:
            scat_species_profile = np.interp(
                background_atm.grids[1], pressure_profile, scat_species_profile
            )

        data2[-1, :, 0, 0] = scat_species_profile

    background_atm.data = data2

    return background_atm
