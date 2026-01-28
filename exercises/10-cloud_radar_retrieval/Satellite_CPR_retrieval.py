#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@created on Wed 7 Jan 2026
@author: Manfred Brath
"""


import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from pyarts import xml


import radar_module as rm


# =============================================================================
# %% paths and constants
# =============================================================================

# define sensor positions and line of sight
# we assume a HALO like airplane with a sensor at 15 km altitude and a line of sight of 180 degrees
min_range_bin_altitude = 2000.0
max_range_bin_altitude = 20000.0
sensor_los = 180.0

# =============================================================================
# %%  load observations and background atmosphere

observation = xr.load_dataset("observation/simulated_CPR_observation.nc")
background_atm = xml.load("observation/background_state.xml")

true_atmosphere = xml.load("atmosphere/atmospheres_true.xml")

# %% plot observation

fig, ax = plt.subplots(1, 1, figsize=(29.7 / 2.54, 10.45 / 2.54))

pcm = ax.pcolormesh(
    observation["latitudes"],
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
ax.set_title("CPR Observation")
cbar = fig.colorbar(pcm, ax=ax, label="Radar Reflectivity [dBZe]")

rm.os.makedirs("plots", exist_ok=True)
fig.savefig("plots/CPR_observation.pdf")

# =============================================================================
# %% prepare a priori atmosphere for retrieval

z_cpr = observation["range_bins"]
z_background = background_atm.get("z", keep_dims=False)
T_background = background_atm.get("T", keep_dims=False)


IWC_apr, RWC_apr = rm.radar_reflectivity_to_apriori(
    observation["Z_raw"].values,
    observation["range_bins"].values,
    z_background,
    4000,
    7000,
    
    )


# =============================================================================
# %% plot apriori hydrometeor profiles for full flight track

fig, ax = plt.subplots(
    2, 1, figsize=(29.7 / 2.54, 20.9 / 2.54), sharex=True, sharey=True
)
ax = ax.flatten()

ax[0].pcolormesh(
    observation["latitudes"],
    z_background / 1000,
    np.log10(IWC_apr),
    shading="auto",
    # norm=rm.matplotlib.colors.LogNorm(vmin=1e-7, vmax=1e-2),
    vmin=-7,
    vmax=-2,
    rasterized=True,
    cmap="inferno",
)
ax[0].set_ylabel("Altitude [km]")
ax[0].set_xlabel("Latitude [deg]")
ax[0].set_title("A Priori SWC-mass_density")
cbar = fig.colorbar(ax[0].collections[0], ax=ax[0], label=r"$10^x$ [kg/m3]")

ax[1].pcolormesh(
    observation["latitudes"],
    z_background / 1000,
    np.log10(RWC_apr),
    shading="auto",
    # norm=rm.matplotlib.colors.LogNorm(vmin=1e-7, vmax=1e-2),
    vmin=-7,
    vmax=-2,
    rasterized=True,
    cmap="inferno",
)
ax[1].set_ylabel("Altitude [km]")
ax[1].set_xlabel("Latitude [deg]")
ax[1].set_title("A Priori RWC-mass_density")
cbar = fig.colorbar(ax[1].collections[0], ax=ax[1], label=r"$10^x$ [kg/m3]")
fig.tight_layout()

rm.os.makedirs("plots", exist_ok=True)
fig.savefig("plots/apriori_hydrometeor_profiles_full_flight.pdf")


# =============================================================================
# %% do the retrieval


retrieval = rm.RadarSimulator()
retrieval.set_frequency_grid(observation["frequencies"].values)

# retrieval.set_liquid_hydrometeor("RWC","mass_density")
retrieval.set_retrieval_quantity(
    "RWC", "mass_density", "AbelBoutle12", "H2O_liquid_full_spectrum"
)
retrieval.set_retrieval_quantity(
    "FWC", "mass_density", "McFarquaharHeymsfield97", "GemSnow_Id32.scat_data"
)
retrieval.prepare_sensor(unit="dBZe")

retrieval_quantities = retrieval.retrieval_quantities


S_y = np.diag(
    np.ones((len(observation["range_bins"])) * len(observation["frequencies"])) * 1.0
)
S_a = [np.diag(np.ones_like(z_background) * 9), np.diag(np.ones_like(z_background) * 9)]


retrieval_results = {
    "z": np.zeros((len(z_background), len(observation["latitudes"]))),
    "Z_raw_fit": np.zeros(
        (
            (len(observation["range_bins"])) * len(observation["frequencies"]),
            len(observation["latitudes"]),
        )
    ),
}
for rq in retrieval_quantities:
    retrieval_results[rq] = np.zeros((len(z_background), len(observation["latitudes"])))
    retrieval_results[f"delta_{rq}"] = np.zeros(
        (len(z_background), len(observation["latitudes"]))
    )


for atm_idx in range(len(observation["latitudes"])):

    print(f"Retrieving atmosphere {atm_idx+1}/{len(observation['latitudes'])}")

    background_atm.set(
        "scat_species-FWC-mass_density",
        IWC_apr[:, atm_idx].reshape((1, len(IWC_apr), 1, 1)),
    )

    background_atm.set(
        "scat_species-RWC-mass_density",
        RWC_apr[:, atm_idx].reshape((1, len(RWC_apr), 1, 1)),
    )

    y_cpr = observation["Z_raw"][:, atm_idx]

    hyd_ret, DeltaHyd, y_fit, full_result = retrieval.hydrometeor_retrieval(
        y_cpr,
        S_y,
        S_a,
        background_atm,
        max_iter=50,
        N_range_bins_edges=len(observation["range_bins"]) + 1,
        min_range_bin_altitude=min_range_bin_altitude,
        max_range_bin_altitude=max_range_bin_altitude,
        Verbosity=True,
        lm_ga_settings=[1e2, 2, 3, 1e2, 1, 99],
        stop_dx=0.01,
    )

    retrieval_results["z"][:, atm_idx] = z_background
    for rq in retrieval_quantities:
        retrieval_results[rq][:, atm_idx] = hyd_ret[rq]
        retrieval_results[f"delta_{rq}"][:, atm_idx] = DeltaHyd[rq]
    retrieval_results["Z_raw_fit"][:, atm_idx] = y_fit


# %% Plot observation, fit, and residuals

fig, ax = plt.subplots(3, 1, figsize=(29.7 / 2.54, 30 / 2.54), sharex=True, sharey=True)
ax = ax.flatten()

pcm0 = ax[0].pcolormesh(
    observation["latitudes"],
    observation["range_bins"] / 1000,
    observation["Z_raw"],
    shading="auto",
    vmin=-35,
    vmax=20,
    rasterized=True,
    cmap="inferno",
)
ax[0].set_ylabel("Altitude [km]")
ax[0].set_xlabel("Latitude [deg]")
ax[0].set_title("CPR Observation")
cbar0 = fig.colorbar(pcm0, ax=ax[0], label="Radar Reflectivity [dBZe]")

pcm1 = ax[1].pcolormesh(
    observation["latitudes"],
    observation["range_bins"] / 1000,
    retrieval_results["Z_raw_fit"] - observation["Z_raw"],
    shading="auto",
    vmin=-1,
    vmax=1,
    rasterized=True,
    cmap="RdBu",
)
ax[1].set_ylabel("Altitude [km]")
ax[1].set_xlabel("Latitude [deg]")
ax[1].set_title("Retrieval Residuals")
cbar1 = fig.colorbar(pcm1, ax=ax[1], label="Radar Reflectivity Residuals [dBZe]")


pcm2 = ax[2].pcolormesh(
    observation["latitudes"],
    observation["range_bins"] / 1000,
    retrieval_results["Z_raw_fit"],
    shading="auto",
    vmin=-35,
    vmax=20,
    rasterized=True,
    cmap="inferno",
)
ax[2].set_ylabel("Altitude [km]")
ax[2].set_xlabel("Latitude [deg]")
ax[2].set_title("Fitted CPR Observation")
cbar2 = fig.colorbar(pcm2, ax=ax[2], label="Radar Reflectivity [dBZe]")


rm.os.makedirs("figures", exist_ok=True)
fig.savefig("figures/cpr_observation_fit_residuals.pdf")

# %% plot retrieved hydrometeor profiles and errors for full flight track

fig, ax = plt.subplots(
    len(retrieval_quantities),
    2,
    figsize=(29.7 / 2.54, 20.9 / 2.54),
    sharex=True,
    sharey=True,
)
ax = ax.flatten()

for i, rq in enumerate(retrieval_quantities):

    pcm = ax[2 * i].pcolormesh(
        observation["latitudes"],
        retrieval_results["z"][:, 0] / 1000,
        retrieval_results[rq],
        shading="auto",
        # norm=rm.matplotlib.colors.LogNorm(vmin=1e-7, vmax=1e-2),
        vmin=-7,
        vmax=-2,
        rasterized=True,
        cmap="inferno",
    )
    ax[2 * i].set_ylabel("Altitude [km]")
    ax[2 * i].set_xlabel("Latitude [deg]")
    ax[2 * i].set_title(f"Retrieved {rq}")
    cbar = fig.colorbar(pcm, ax=ax[2 * i], label=r"$10^x$ [kg/m3]")

    pcm = ax[2 * i + 1].pcolormesh(
        observation["latitudes"],
        retrieval_results["z"][:, 0] / 1000,
        retrieval_results[f"delta_{rq}"],
        shading="auto",
        vmin=-0.5,
        vmax=0.5,
        rasterized=True,
        cmap="RdBu",
    )
    ax[2 * i + 1].set_ylabel("Altitude [km]")
    ax[2 * i + 1].set_xlabel("Latitude [deg]")
    ax[2 * i + 1].set_title(f"Retrieval Uncertainty of {rq}")
    rqq = rq.split("-")[0]
    cbar = fig.colorbar(
        pcm, ax=ax[2 * i + 1], label=r"$\log_{10}$" + f" relative error of {rqq}"
    )

fig.tight_layout()
rm.os.makedirs("plots", exist_ok=True)
fig.savefig("plots/retrieved_hydrometeor_profiles_full_flight.pdf")


# %% plot

# calculate total ice and total liquid
TLWC = np.zeros_like(RWC_apr)
TFWC = np.zeros_like(IWC_apr)

for i in range(len(true_atmosphere)):

    TLWC[:, i] = true_atmosphere[i].get(
        "scat_species-LWC-mass_density", keep_dims=False
    ) + true_atmosphere[i].get("scat_species-RWC-mass_density", keep_dims=False)

    TFWC[:, i] = (
        true_atmosphere[i].get("scat_species-IWC-mass_density", keep_dims=False)
        + true_atmosphere[i].get("scat_species-SWC-mass_density", keep_dims=False)
        + true_atmosphere[i].get("scat_species-GWC-mass_density", keep_dims=False)
        + true_atmosphere[i].get("scat_species-HWC-mass_density", keep_dims=False)
    )

TLWC[TLWC < 1e-18] = 1e-18
TFWC[TFWC < 1e-18] = 1e-18


fig, ax = plt.subplots(
    2, 3, figsize=(29.7 / 2.54, 20.9 / 2.54), sharex=True, sharey=True
)
ax = ax.flatten()

ax[0].pcolormesh(
    observation["latitudes"],
    retrieval_results["z"][:, 0] / 1000,
    np.log10(TLWC),
    shading="auto",
    vmin=-7,
    vmax=-2,
    rasterized=True,
    cmap="inferno",
)
ax[0].set_ylabel("Altitude [km]")
ax[0].set_xlabel("Latitude [deg]")
ax[0].set_title("True Total Liquid Water Content")
cbar0 = fig.colorbar(ax[0].collections[0], ax=ax[0], label=r"$10^x$ [kg/m3]")

ax[1].pcolormesh(
    observation["latitudes"],
    retrieval_results["z"][:, 0] / 1000,
    retrieval_results["RWC-mass_density"] - np.log10(TLWC),
    shading="auto",
    vmin=-1,
    vmax=1,
    rasterized=True,
    cmap="RdBu",
)
ax[1].set_ylabel("Altitude [km]")
ax[1].set_xlabel("Latitude [deg]")
ax[1].set_title("Retrieval Error Total Liquid Water Content")
cbar1 = fig.colorbar(
    ax[1].collections[0], ax=ax[1], label=r"$\log_{10}$ relative error"
)


ax[2].pcolormesh(
    observation["latitudes"],
    retrieval_results["z"][:, 0] / 1000,
    retrieval_results["RWC-mass_density"],
    shading="auto",
    vmin=-7,
    vmax=-2,
    rasterized=True,
    cmap="inferno",
)
ax[2].set_ylabel("Altitude [km]")
ax[2].set_xlabel("Latitude [deg]")
ax[2].set_title("Retrieved Liquid Water Content")
cbar2 = fig.colorbar(ax[2].collections[0], ax=ax[2], label=r"$10^x$ [kg/m3]")

ax[3].pcolormesh(
    observation["latitudes"],
    retrieval_results["z"][:, 0] / 1000,
    np.log10(TFWC),
    shading="auto",
    vmin=-7,
    vmax=-2,
    rasterized=True,
    cmap="inferno",
)
ax[3].set_ylabel("Altitude [km]")
ax[3].set_xlabel("Latitude [deg]")
ax[3].set_title("True Total Frozen Water Content")
cbar3 = fig.colorbar(ax[3].collections[0], ax=ax[3], label=r"$10^x$ [kg/m3]")

ax[4].pcolormesh(
    observation["latitudes"],
    retrieval_results["z"][:, 0] / 1000,
    retrieval_results["FWC-mass_density"] - np.log10(TFWC),
    shading="auto",
    vmin=-1,
    vmax=1,
    rasterized=True,
    cmap="RdBu",
)
ax[4].set_ylabel("Altitude [km]")
ax[4].set_xlabel("Latitude [deg]")
ax[4].set_title("Retrieval Error Total Frozen Water Content")
cbar4 = fig.colorbar(
    ax[4].collections[0], ax=ax[4], label=r"$\log_{10}$ relative error"
)

ax[5].pcolormesh(
    observation["latitudes"],
    retrieval_results["z"][:, 0] / 1000,
    retrieval_results["FWC-mass_density"],
    shading="auto",
    vmin=-7,
    vmax=-2,
    rasterized=True,
    cmap="inferno",
)
ax[5].set_ylabel("Altitude [km]")
ax[5].set_xlabel("Latitude [deg]")
ax[5].set_title("Retrieved Frozen Water Content")
cbar5 = fig.colorbar(ax[5].collections[0], ax=ax[5], label=r"$10^x$ [kg/m3]")
fig.tight_layout()
rm.os.makedirs("plots", exist_ok=True)
fig.savefig("plots/retrieved_total_liquid_frozen_water_content_full_flight.pdf")

