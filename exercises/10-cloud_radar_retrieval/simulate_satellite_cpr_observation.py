#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@created on Wed 7 Jan 2026
@author: Manfred Brath
"""
from copy import deepcopy
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pyarts import xml

import radar_module as rm


# =============================================================================
# %% paths and constants
# =============================================================================


# atmospheric data
atms = xml.load("atmosphere/atmospheres_true.xml")
auxs = xml.load("atmosphere/aux1d_true.xml")

# surface reflectivity
# It is assumed that the surface reflectivity is known and constant
surface_reflectivity = 0.4

# surface temperature
# It is assumed that the surface temperature is known and constant
surface_temperature = 300.0  # K

# define sensor positions and line of sight
# we assume a Cloudsat like sensor with line of sight of 180 degrees
min_range_bin_altitude = 2000.0
max_range_bin_altitude = 20000.0
sensor_los = 180.0

# latitude
lat = np.array([auxs[i].data[0] for i in range(len(auxs))])
lon = np.array([auxs[i].data[1] for i in range(len(auxs))])

f_grid = [90e9]

N_range_bins = 73


background_atm_index = 450

# =============================================================================
# %% simulate the observation

cpr = rm.RadarSimulator()
cpr.set_frequency_grid(f_grid)
cpr.scattering_data_from_arts_xml_package = False
cpr.set_paths(rm.os.path.join(rm.os.getcwd(), "scattering_data"))
cpr.define_mie_MilbrandtYau_scheme()
cpr.prepare_sensor(unit="dBZe")

observation = {
    "Z": np.zeros((len(f_grid), N_range_bins - 1, len(atms))),
    "Z_raw": np.zeros(((N_range_bins - 1) * len(f_grid), len(atms))),
    "range_bins": np.zeros((N_range_bins - 1,)),
    "frequencies": f_grid,
    "latitudes": lat,
    "longitudes": lon,
    "range_bin_units": "m",
    "Z_units": "dBZe",
    "frequency_units": "Hz",
    "latitude_units": "deg",
    "longitude_units": "deg",
}

for atm_idx in range(len(atms)):

    print(f"Simulating observation for atmosphere {atm_idx+1}/{len(atms)}")

    # with jacobian
    result_i = cpr.cloud_radar_1D(
        atm=atms[atm_idx],
        min_range_bin_altitude=min_range_bin_altitude,
        max_range_bin_altitude=max_range_bin_altitude,
        N_range_bins_edges=N_range_bins,
        retrieval_quantities=[],
    )

    observation["Z"][:, :, atm_idx] = result_i["Z"]
    observation["Z_raw"][:, atm_idx] = result_i["Z_raw"]

    if atm_idx == 0:
        observation["range_bins"] = result_i["range_bins"][:]

# %% plot observation

fig, ax = plt.subplots(1, 1, figsize=(29.7 / 2.54, 10.45 / 2.54))

pcm = ax.pcolormesh(
    lat,
    observation["range_bins"] / 1000,
    observation["Z_raw"],
    shading="auto",
    vmin=-35,
    vmax=20,
    rasterized=True,
    cmap="inferno",
)
ax.set_ylabel("Altitude [km]")
ax.set_xlabel("Latitude [deg]")
ax.set_title("Simulated CPR Observation")
cbar = fig.colorbar(pcm, ax=ax, label="Radar Reflectivity [dBZe]")

rm.os.makedirs("check_plots", exist_ok=True)
fig.savefig("figures/simulated_CPR_observation.pdf")


# =============================================================================
# %% export observation data as netcdf file

ds = xr.Dataset(
    {
        "Z": (
            ("frequency", "range_bin", "profile"),
            observation["Z"],
            {"units": "dBZe"},
        ),
        "Z_raw": (
            ("range_bin_frequency", "profile"),
            observation["Z_raw"],
            {"units": "dBZe"},
        ),
        "range_bins": (("range_bin",), observation["range_bins"], {"units": "m"}),
        "frequencies": (("frequency",), observation["frequencies"], {"units": "Hz"}),
        "latitudes": (("profile",), observation["latitudes"], {"units": "deg"}),
        "longitudes": (("profile",), observation["longitudes"], {"units": "deg"}),
    },
    coords={
        "range_bin_frequency": np.repeat(
            observation["range_bins"],
            len(observation["frequencies"]),
        ),
        "frequency": observation["frequencies"],
        "range_bin": observation["range_bins"],
        "profile": np.arange(len(atms)),
    },
)

ds.to_netcdf("observation/simulated_CPR_observation.nc")


# =============================================================================
# %% prepare background atmosphere

background_atm = deepcopy(atms[background_atm_index])


# variables=background_atm.grids[0]
# variables.append('scat_species-FWC-mass_density')
# background_atm.set_grid(0,variables)
# data=background_atm.data.value
# data2=np.concatenate((data,np.zeros((1,np.size(data,1),1,1))),axis=0)
# background_atm.data=data2

# def add_scat_species(background_atm, species_name, type_name):


#     variables = background_atm.grids[0]
#     variables.append(f'scat_species-{species_name}-{type_name}')
#     background_atm.set_grid(0, variables)
#     data = background_atm.data.value
#     data2 = np.concatenate((data, np.zeros((1, np.size(data, 1), 1, 1))), axis=0)
#     background_atm.data = data2

#     return background_atm
# #

background_atm = rm.add_scat_species(background_atm, "FWC", "mass_density")

background_atm.savexml("observation/background_state.xml")
