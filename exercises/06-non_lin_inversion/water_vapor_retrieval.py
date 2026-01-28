#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temperature Retrieval Analysis Script

This script performs temperature profile retrievals from radiometric observations
in the 183 GHz range using the Optimal Estimation Method (OEM).

The script:
1. Loads atmospheric data (true profiles, a priori data from dropsondes)
2. Loads observation data (frequencies, measurements, location)
3. Performs single profile retrieval with diagnostics
4. Performs retrieval for a complete observation segment
5. Creates visualization plots of the results

Key Features:
- Single profile retrieval with averaging kernels and gain matrix analysis
- Multiple profile retrieval along a latitude segment
- Comparison between retrieved, true and a priori profiles
- Uncertainty estimation and visualization
- Observation fitting analysis and residuals

Required Data Files:
- atmosphere/atmospheres_true.xml: True atmospheric profiles
- observation/dropsonde.xml: A priori and background data
- observation/SensorCharacteristics_183GHz.xml: Sensor characteristics
- observation/y_obs_183GHz.xml: Observation vector
- observation/lat.xml: Latitude information

Dependencies:
    numpy
    matplotlib
    pyarts
    nonlin_oem

@author: Manfred Brath
Created: Tue Jan 7 12:13:59 2025

"""

import numpy as np
import matplotlib.pyplot as plt

import pyarts as pa
import nonlin_oem as nlo


# =============================================================================
# %% read data
# =============================================================================

# surface temperature
surface_temperature = 300  # K

# surface reflectivity
surface_reflectivity = 0.4

# sensor position and line of sight
sensor_pos = 15e3
sensor_los = 180.0

# load true data
atms = pa.xml.load("atmosphere/atmospheres_true.xml")

# load dropsonde data this will serve as a priori and background data
dropsonde = pa.xml.load("observation/dropsonde_WV.xml")

# load frequency data for 183 GHz channels
sensor_characteristics=pa.xml.load("observation/SensorCharacteristics_183GHz.xml")[:]

# load measurement vector
y_obs = pa.xml.load("observation/y_obs_183GHz.xml")

lat = pa.xml.load("observation/lat.xml")

NeDT = 0.6  # K

# %% do the retrieval for one observation

# select obervation idx
idx = 328
y = y_obs[idx, :]

# create and add a priori covariance matrix
correlation_length = nlo.set_correlation_length(
    dropsonde.get("z", keep_dims=False), len_sfc=2000, len_toa=2000
)
z_dropsonde=dropsonde.get("z", keep_dims=False)

# define a priori uncertainty according to Prange et al. 2021 (AMT)
delta_x=np.ones_like(z_dropsonde)
z_idx=np.where(z_dropsonde<2000)
delta_x[z_idx]=9./20000.*z_dropsonde[z_idx]+0.1
delta_x*=1.

S_a = nlo.create_apriori_covariance_matrix(
    dropsonde.get("abs_species-H2O", keep_dims=False),
    dropsonde.get("z", keep_dims=False),
    delta_x,
    correlation_length,
)

# Define Se and its invers
S_y = pa.arts.Sparse(np.diag(np.ones(np.size(sensor_characteristics,0)) * NeDT))

#Temperature retrieval for selected observation
WV_ret, DeltaWV, y_fit, A, G, result = nlo.watervapor_retrieval(
    y,
    [],
    sensor_pos,
    sensor_los,
    dropsonde,
    surface_temperature,
    surface_reflectivity,
    S_y,
    S_a,
    Diagnostics=True,
    Verbosity=True,
    sensor_description=sensor_characteristics
)


# %% plot results for the single observation

WV_apr = np.log10(dropsonde.get("abs_species-H2O", keep_dims=False))
WV_true = np.log10(atms[idx].get("abs_species-H2O", keep_dims=False))
z_true = atms[idx].get("z", keep_dims=False)
z_ret = dropsonde.get("z", keep_dims=False)
DeltaWV_apriori = np.sqrt(np.diag(S_a))

# Frequencies of the channels
f_grid=sensor_characteristics[:,0]+sensor_characteristics[:,1]+sensor_characteristics[:,2]

fig, ax = plt.subplots(1, 3, figsize=(14.14, 10))

#plot difference to true temperature
ax[0].plot(WV_ret - WV_true, z_ret, "s-", color="r", label="ret")
ax[0].fill_betweenx(
    z_ret, WV_ret - WV_true - DeltaWV, WV_ret - WV_true + DeltaWV, alpha=0.3, color="r"
)
ax[0].plot(WV_apr - WV_true, z_ret, "o-", color="g", label="apriori")
ax[0].fill_betweenx(
    z_ret,
    WV_apr - WV_true - DeltaWV_apriori,
    WV_apr - WV_true + DeltaWV_apriori,
    alpha=0.3,
    color="g",
)
ax[0].set_title("Difference to T$_{true}$")
ax[0].set_xlabel("log vmr")
ax[0].set_ylabel("Altitude [m]")
ax[0].legend()

# Plot true temperature profile
ax[1].semilogx(10**WV_true, z_true, "x-", label="truth")
ax[1].semilogx(10**WV_apr, z_ret, "x-", label="a priori")
ax[1].semilogx(10**WV_ret, z_ret, "x-", label="retrieved")
ax[1].set_xlabel("vmr true [K]")
ax[1].set_ylabel("Altitude [m]")
ax[1].legend()


ax[2].plot(f_grid, y, "x-", label="obs")
ax[2].plot(f_grid, y_fit, "s-", label="fit")
ax[2].set_xlabel("Frequency [GHz]")
ax[2].set_ylabel("Brightness temperature [K]")
ax[2].legend()
fig.tight_layout()


#new figure
fig2, ax2 = plt.subplots(1, 2, figsize=(14.14, 10))

#plot averaging kernels
colormap = plt.cm.YlOrRd
colors = [colormap(i) for i in np.linspace(0, 1, np.size(A, 1))]
for i in range(np.size(A, 1)):
    ax2[0].plot(
        A[:, i],
        z_ret,
        color=colors[i],
    )
A_sum = np.sum(A, axis=1)
ax2[0].plot(A_sum / 5, z_ret, color="k", label="$\Sigma A_{i}$/5")
ax2[0].legend()
ax2[0].set_xlabel("Altitude [m]")
ax2[0].set_ylabel("Altitude [m]")
ax2[0].set_title("Averaging Kernels")


#plot gain matrix
for i in range(np.size(G, 1)):
    ax2[1].plot(G[:, i], z_ret, label=f"{f_grid[i]/1e9:.2f}GHz")
ax2[1].set_title("Gain Matrix")
ax2[1].legend()
ax2[1].set_xlabel("Contributions / KK$^{-1}$")
ax2[1].set_ylabel("Altitude [m]")
fig.tight_layout()


# %% now do the retrieval for the whole segment

#allocate
DeltaWV_all = np.zeros((np.size(y_obs, 0), dropsonde.get("z", keep_dims=False).size))
WV_all = np.zeros((np.size(y_obs, 0), dropsonde.get("z", keep_dims=False).size))
y_fit_all = np.zeros((np.size(y_obs, 0), len(f_grid)))
A_summed_all = np.zeros((np.size(y_obs, 0), dropsonde.get("z", keep_dims=False).size))
G_summed_all = np.zeros((np.size(y_obs, 0), dropsonde.get("z", keep_dims=False).size))


#loop over the observations
for i in range(np.size(y_obs, 0)):

    print(f"...retrieving profile {i} of {np.size(y_obs, 0)}")

    y = y_obs[i, :]
    WV_ret, DeltaWV, y_fit, A, G, _ = nlo.watervapor_retrieval(
        y,
        [],
        sensor_pos,
        sensor_los,
        dropsonde,
        surface_temperature,
        surface_reflectivity,
        S_y,
        S_a,
        Diagnostics=True,
        sensor_description=sensor_characteristics
    )


    DeltaWV_all[i, :] = DeltaWV
    WV_all[i, :] = WV_ret
    y_fit_all[i, :] = y_fit
    A_summed_all[i, :] = np.sum(A, axis=1)
    G_summed_all[i, :] = np.sum(G, axis=1)



# %% plot the whole segment

fig, ax = plt.subplots(2, 3, figsize=(14.14, 10))

#plot difference to a priori
cmap = "YlOrRd"
data = WV_all - WV_apr  # T_all.mean(0)
data_max = np.max(np.abs(data))
pcm = ax[0, 0].pcolormesh(
    lat, z_ret / 1e3, data.T, cmap="RdBu_r", clim=[-data_max, data_max], rasterized=True
)
fig.colorbar(pcm, ax=ax[0, 0], label="log$_{10}vmr_{ret}$ - log$_{10}vmr_{apr}$ / 1")
ax[0, 0].set_xlabel("Latitude [deg]")
ax[0, 0].set_ylabel("Altitude [km]")
ax[0, 0].set_title("Difference to a priori")

data = DeltaWV_all
pcm = ax[0, 1].pcolormesh(
    lat,
    z_ret / 1e3,
    data.T,
    cmap=cmap,
    clim=[np.min(data), np.max(data)],
    rasterized=True,
)

#plot retrieved water vapor uncertainty
fig.colorbar(pcm, ax=ax[0, 1], label="$\Delta$ log$_{10}vmr$ / 1")
ax[0, 1].set_xlabel("Latitude [deg]")
ax[0, 1].set_ylabel("Altitude [km]")
ax[0, 1].set_title("Retrieved log$_{10}vmr$ uncertainty")

#plot observed and fitted brightness temperatures
for i in range(np.size(y_obs, 1)):
    ax[1, 0].plot(lat, y_obs[:, i], label=f"{f_grid[i]/1e9:.2f}$\,$GHz, obs")
ax[1, 0].set_prop_cycle(None)
for i in range(np.size(y_fit_all, 1)):
    ax[1, 0].plot(
        lat, y_fit_all[:, i], "--"
    )
ax[1, 0].set_prop_cycle(None)
ax[1, 0].set_xlabel("Latitude [deg]")
ax[1, 0].set_ylabel("Brightness temperature [K]")
ax[1, 0].legend()

#plot residuals
ax[1, 1].plot(lat, y_obs - y_fit_all, "-", label="residual")
ax[1, 1].set_xlabel("Latitude [deg]")
ax[1, 1].set_ylabel("Brightness temperature [K]")
ax[1, 1].set_title("Residuals")

#plot retrieved data
pcm = ax[0, 2].pcolormesh(
    lat,
    z_ret / 1e3,
    WV_all.T,
    cmap=cmap,
    clim=[np.min(WV_all), np.max(WV_all)],
    rasterized=True,
)
fig.colorbar(pcm, ax=ax[0, 2], label="log$_{10}vmr$ / 1")
ax[0, 2].set_xlabel("Latitude [deg]")
ax[0, 2].set_ylabel("Altitude [km]")
ax[0, 2].set_title("Retrieved log$_{10}vmr$")

#plot true data
pcm = ax[1, 2].pcolormesh(
    lat,
    z_true / 1e3,
    np.log10(
        np.array([atms[i].get("abs_species-H2O", keep_dims=False) for i in range(len(atms))]).T
    ),
    cmap=cmap,
    clim=[np.min(WV_all), np.max(WV_all)],
    rasterized=True,
)
fig.colorbar(pcm, ax=ax[1, 2], label="log$_{10}vmr$ / 1")
ax[1, 2].set_xlabel("Latitude [deg]")
ax[1, 2].set_ylabel("Altitude [km]")
ax[1, 2].set_title("True log$_{10}vmr$")    

fig.tight_layout()
