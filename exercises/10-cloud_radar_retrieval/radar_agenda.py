#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Radar Agenda Module for ARTS Cloud Radar Retrieval

This module provides particle number density (PND) agendas for various microphysics
schemes used in cloud radar retrievals with ARTS (Atmospheric Radiative Transfer
Simulator).

The module implements three main microphysics schemes:
1. Seifert and Beheng (2006) two-moment scheme (SB06)
2. Milbrandt and Yau (2005) two-moment scheme (MY05)
3. COSMO-style scheme (CG)

Each scheme provides PND agendas for different hydrometeor types:
- LWC: Liquid Water Content (cloud water)
- IWC: Ice Water Content (cloud ice)
- RWC: Rain Water Content
- SWC: Snow Water Content
- GWC: Graupel Water Content
- HWC: Hail Water Content

Additional PSD (Particle Size Distribution) agendas are also provided:
- Delanoë et al. (2014)
- Abel and Boutle (2012)
- Field et al. (2007) - Midlatitude and Tropical regimes
- McFarquahar and Heymsfield (1997)

Functions:
    create_pnd_agendas_SB06_in_WS: Create SB06 PND agendas in workspace
    set_pnd_agendas_SB06: Set SB06 PND agendas to workspace variables
    create_pnd_agendas_MY05_in_WS: Create MY05 PND agendas in workspace
    set_pnd_agendas_MY05: Set MY05 PND agendas to workspace variables
    create_pnd_agendas_CG_in_WS: Create COSMO-style PND agendas in workspace
    set_pnd_agendas_CG: Set COSMO-style PND agendas to workspace variables
    create_additional_pnd_agendas_in_WS: Create additional PND agendas in workspace
    set_additional_pnd_agendas: Set additional PND agendas to workspace variables
    create_agendas_in_ws: Create all PND agendas and return sorted list of names
    iy_radar_agenda_singlepol: Single polarization radar intensity agenda

Notes:
    - The COSMO-style scheme PSDs are not exactly those from COSMO model
    - Graupel and snow are similar to COSMO graupel
    - Cloud ice and cloud liquid are from Geer and Baordo (2014)
    - RWC uses an existing ARTS parameterization
'''
    
from pyarts.workspace import arts_agenda

# =============================================================================
# pnd_agendas
# =============================================================================

#### Seifert and Beheng, 2006 two moment scheme  ####
# ==========================================================================


def create_pnd_agendas_SB06_in_WS(ws):

    pnd_agenda_list = [
        "pnd_agenda_SB06LWC",
        "pnd_agenda_SB06IWC",
        "pnd_agenda_SB06RWC",
        "pnd_agenda_SB06SWC",
        "pnd_agenda_SB06GWC",
        "pnd_agenda_SB06HWC",
    ]

    for pnd_agenda in pnd_agenda_list:
        ws.AgendaCreate(pnd_agenda)

    return ws, pnd_agenda_list


# LWC
@arts_agenda
def pnd_agenda_SB06LWC(ws):
    ws.ExtractFromMetaSingleScatSpecies(
        ws.pnd_size_grid, ws.scat_meta, "mass", ws.agenda_array_index
    )
    ws.Copy(ws.psd_size_grid, ws.pnd_size_grid)
    ws.psdSeifertBeheng06(hydrometeor_type="cloud_water", t_min=0, t_max=999, picky=0)
    ws.pndFromPsdBasic()


# IWC
@arts_agenda
def pnd_agenda_SB06IWC(ws):
    ws.ExtractFromMetaSingleScatSpecies(
        ws.pnd_size_grid, ws.scat_meta, "mass", ws.agenda_array_index
    )
    ws.Copy(ws.psd_size_grid, ws.pnd_size_grid)
    ws.psdSeifertBeheng06(hydrometeor_type="cloud_ice", t_min=0, t_max=999, picky=0)
    ws.pndFromPsdBasic()


# Rain
@arts_agenda
def pnd_agenda_SB06RWC(ws):
    ws.ExtractFromMetaSingleScatSpecies(
        ws.pnd_size_grid, ws.scat_meta, "mass", ws.agenda_array_index
    )
    ws.Copy(ws.psd_size_grid, ws.pnd_size_grid)
    ws.psdSeifertBeheng06(hydrometeor_type="rain", t_min=0, t_max=999, picky=0)
    ws.pndFromPsdBasic()


# Snow
@arts_agenda
def pnd_agenda_SB06SWC(ws):
    ws.ExtractFromMetaSingleScatSpecies(
        ws.pnd_size_grid, ws.scat_meta, "mass", ws.agenda_array_index
    )
    ws.Copy(ws.psd_size_grid, ws.pnd_size_grid)
    ws.psdSeifertBeheng06(hydrometeor_type="snow", t_min=0, t_max=999, picky=0)
    ws.pndFromPsdBasic()


# Graupel
@arts_agenda
def pnd_agenda_SB06GWC(ws):
    ws.ExtractFromMetaSingleScatSpecies(
        ws.pnd_size_grid, ws.scat_meta, "mass", ws.agenda_array_index
    )
    ws.Copy(ws.psd_size_grid, ws.pnd_size_grid)
    ws.psdSeifertBeheng06(hydrometeor_type="graupel", t_min=0, t_max=999, picky=0)
    ws.pndFromPsdBasic()


# Hail
@arts_agenda
def pnd_agenda_SB06HWC(ws):
    ws.ExtractFromMetaSingleScatSpecies(
        ws.pnd_size_grid, ws.scat_meta, "mass", ws.agenda_array_index
    )
    ws.Copy(ws.psd_size_grid, ws.pnd_size_grid)
    ws.psdSeifertBeheng06(hydrometeor_type="hail", t_min=0, t_max=999, picky=0)
    ws.pndFromPsdBasic()


def set_pnd_agendas_SB06(ws):
    ws.pnd_agenda_SB06LWC = pnd_agenda_SB06LWC
    ws.pnd_agenda_SB06IWC = pnd_agenda_SB06IWC
    ws.pnd_agenda_SB06RWC = pnd_agenda_SB06RWC
    ws.pnd_agenda_SB06SWC = pnd_agenda_SB06SWC
    ws.pnd_agenda_SB06GWC = pnd_agenda_SB06GWC
    ws.pnd_agenda_SB06HWC = pnd_agenda_SB06HWC

    return ws


#### Milbrandt and Yau, 2005 two moment scheme  ####
# ==========================================================================


def create_pnd_agendas_MY05_in_WS(ws):

    pnd_agenda_list = [
        "pnd_agenda_MY05LWC",
        "pnd_agenda_MY05IWC",
        "pnd_agenda_MY05RWC",
        "pnd_agenda_MY05SWC",
        "pnd_agenda_MY05GWC",
        "pnd_agenda_MY05HWC",
    ]

    for pnd_agenda in pnd_agenda_list:
        ws.AgendaCreate(pnd_agenda)

    return ws, pnd_agenda_list


# LWC
@arts_agenda
def pnd_agenda_MY05LWC(ws):
    ws.ExtractFromMetaSingleScatSpecies(
        ws.pnd_size_grid, ws.scat_meta, "diameter_max", ws.agenda_array_index
    )
    ws.Copy(ws.psd_size_grid, ws.pnd_size_grid)
    ws.psdMilbrandtYau05(hydrometeor_type="cloud_water", t_min=0, t_max=999, picky=0)
    ws.pndFromPsdBasic()


# IWC
@arts_agenda
def pnd_agenda_MY05IWC(ws):
    ws.ExtractFromMetaSingleScatSpecies(
        ws.pnd_size_grid, ws.scat_meta, "diameter_max", ws.agenda_array_index
    )
    ws.Copy(ws.psd_size_grid, ws.pnd_size_grid)
    ws.psdMilbrandtYau05(hydrometeor_type="cloud_ice", t_min=0, t_max=999, picky=0)
    ws.pndFromPsdBasic()


# Rain
@arts_agenda
def pnd_agenda_MY05RWC(ws):
    ws.ExtractFromMetaSingleScatSpecies(
        ws.pnd_size_grid, ws.scat_meta, "diameter_max", ws.agenda_array_index
    )
    ws.Copy(ws.psd_size_grid, ws.pnd_size_grid)
    ws.psdMilbrandtYau05(hydrometeor_type="rain", t_min=0, t_max=999, picky=0)
    ws.pndFromPsdBasic()


# Snow
@arts_agenda
def pnd_agenda_MY05SWC(ws):
    ws.ExtractFromMetaSingleScatSpecies(
        ws.pnd_size_grid, ws.scat_meta, "diameter_max", ws.agenda_array_index
    )
    ws.Copy(ws.psd_size_grid, ws.pnd_size_grid)
    ws.psdMilbrandtYau05(hydrometeor_type="snow", t_min=0, t_max=999, picky=0)
    ws.pndFromPsdBasic()


# Graupel
@arts_agenda
def pnd_agenda_MY05GWC(ws):
    ws.ExtractFromMetaSingleScatSpecies(
        ws.pnd_size_grid, ws.scat_meta, "diameter_max", ws.agenda_array_index
    )
    ws.Copy(ws.psd_size_grid, ws.pnd_size_grid)
    ws.psdMilbrandtYau05(hydrometeor_type="graupel", t_min=0, t_max=999, picky=0)
    ws.pndFromPsdBasic()


# Hail
@arts_agenda
def pnd_agenda_MY05HWC(ws):
    ws.ExtractFromMetaSingleScatSpecies(
        ws.pnd_size_grid, ws.scat_meta, "diameter_max", ws.agenda_array_index
    )
    ws.Copy(ws.psd_size_grid, ws.pnd_size_grid)
    ws.psdMilbrandtYau05(hydrometeor_type="hail", t_min=0, t_max=999, picky=0)
    ws.pndFromPsdBasic()


def set_pnd_agendas_MY05(ws):
    ws.pnd_agenda_MY05LWC = pnd_agenda_MY05LWC
    ws.pnd_agenda_MY05IWC = pnd_agenda_MY05IWC
    ws.pnd_agenda_MY05RWC = pnd_agenda_MY05RWC
    ws.pnd_agenda_MY05SWC = pnd_agenda_MY05SWC
    ws.pnd_agenda_MY05GWC = pnd_agenda_MY05GWC
    ws.pnd_agenda_MY05HWC = pnd_agenda_MY05HWC

    return ws


#### Cosmo Style Scheme  ####
# ==========================================================================
"""
The psds are not exactly the ones from the cosmo.
Graupel and snow should be similar to COSMO graupel.
Cloud ice and cloud liquid are taken from Geer and Baordo (2014).
RWC is simply one that was alreay existing in ARTS."""


def create_pnd_agendas_CG_in_WS(ws):

    pnd_agenda_list = [
        "pnd_agenda_CGLWC",
        "pnd_agenda_CGIWC",
        "pnd_agenda_CGRWC",
        "pnd_agenda_CGSWC_tropic",
        "pnd_agenda_CGSWC_midlatitude",
        "pnd_agenda_CGGWC",
    ]

    for pnd_agenda in pnd_agenda_list:
        ws.AgendaCreate(pnd_agenda)

    return ws, pnd_agenda_list


# LWC
@arts_agenda
def pnd_agenda_CGLWC(ws):
    ws.ExtractFromMetaSingleScatSpecies(
        ws.pnd_size_grid, ws.scat_meta, "diameter_max", ws.agenda_array_index
    )
    ws.Copy(ws.psd_size_grid, ws.pnd_size_grid)
    ws.ScatSpeciesSizeMassInfo(x_unit="dmax", species_index=ws.agenda_array_index)
    ws.psdModifiedGammaMass(n0=-999, mu=2, la=2.13e5, ga=1, t_min=0, t_max=400)
    ws.pndFromPsdBasic()


# IWC
@arts_agenda
def pnd_agenda_CGIWC(ws):
    ws.ExtractFromMetaSingleScatSpecies(
        ws.pnd_size_grid, ws.scat_meta, "diameter_max", ws.agenda_array_index
    )
    ws.Copy(ws.psd_size_grid, ws.pnd_size_grid)
    ws.ScatSpeciesSizeMassInfo(x_unit="dmax", species_index=ws.agenda_array_index)
    ws.psdModifiedGammaMass(n0=-999, mu=2, la=2.05e5, ga=1, t_min=0, t_max=400)
    ws.pndFromPsdBasic()


# RWC
@arts_agenda
def pnd_agenda_CGRWC(ws):
    ws.ExtractFromMetaSingleScatSpecies(
        ws.pnd_size_grid, ws.scat_meta, "diameter_max", ws.agenda_array_index
    )
    ws.Copy(ws.psd_size_grid, ws.pnd_size_grid)
    ws.ScatSpeciesSizeMassInfo(x_unit="dmax", species_index=ws.agenda_array_index)
    ws.psdAbelBoutle12(t_min=0, t_max=400)
    ws.pndFromPsdBasic()


# SWC tropic
@arts_agenda
def pnd_agenda_CGSWC_tropic(ws):
    ws.ExtractFromMetaSingleScatSpecies(
        ws.pnd_size_grid, ws.scat_meta, "diameter_max", ws.agenda_array_index
    )
    ws.Copy(ws.psd_size_grid, ws.pnd_size_grid)
    ws.ScatSpeciesSizeMassInfo(x_unit="dmax", species_index=ws.agenda_array_index)
    ws.psdFieldEtAl07(regime="TR", t_min=0, t_max=400)
    ws.pndFromPsdBasic()


# SWC tropic
@arts_agenda
def pnd_agenda_CGSWC_midlatitude(ws):
    ws.ExtractFromMetaSingleScatSpecies(
        ws.pnd_size_grid, ws.scat_meta, "diameter_max", ws.agenda_array_index
    )
    ws.Copy(ws.psd_size_grid, ws.pnd_size_grid)
    ws.ScatSpeciesSizeMassInfo(x_unit="dmax", species_index=ws.agenda_array_index)
    ws.psdFieldEtAl07(regime="ML", t_min=0, t_max=400)
    ws.pndFromPsdBasic()


# GWC
@arts_agenda
def pnd_agenda_CGGWC(ws):
    ws.ExtractFromMetaSingleScatSpecies(
        ws.pnd_size_grid, ws.scat_meta, "diameter_max", ws.agenda_array_index
    )
    ws.Copy(ws.psd_size_grid, ws.pnd_size_grid)
    ws.ScatSpeciesSizeMassInfo(x_unit="dmax", species_index=ws.agenda_array_index)
    ws.psdModifiedGammaMass(n0=4e6, mu=0, la=-999, ga=1, t_min=0, t_max=400)
    ws.pndFromPsdBasic()


def set_pnd_agendas_CG(ws):
    ws.pnd_agenda_CGLWC = pnd_agenda_CGLWC
    ws.pnd_agenda_CGIWC = pnd_agenda_CGIWC
    ws.pnd_agenda_CGRWC = pnd_agenda_CGRWC
    ws.pnd_agenda_CGSWC_tropic = pnd_agenda_CGSWC_tropic
    ws.pnd_agenda_CGSWC_midlatitude = pnd_agenda_CGSWC_midlatitude
    ws.pnd_agenda_CGGWC = pnd_agenda_CGGWC

    return ws


# =============================================================================
# additional pnd agendas


def create_additional_pnd_agendas_in_WS(ws):

    additional_pnd_agenda_list = [
        "pnd_agenda_Delanoe14",
        "pnd_agenda_AbelBoutle12",
        "pnd_agenda_FieldEtAl07ML",
        "pnd_agenda_FieldEtAl07TR",
        "pnd_agenda_McFarquaharHeymsfield97",
    ]

    for pnd_agenda in additional_pnd_agenda_list:
        ws.AgendaCreate(pnd_agenda)

    return ws, additional_pnd_agenda_list


@arts_agenda
def pnd_agenda_Delanoe14(ws):
    ws.ExtractFromMetaSingleScatSpecies(
        ws.pnd_size_grid, ws.scat_meta, "diameter_volume_equ", ws.agenda_array_index
    )
    ws.Copy(ws.psd_size_grid, ws.pnd_size_grid)
    ws.ScatSpeciesSizeMassInfo(x_unit="dveq", species_index=ws.agenda_array_index)
    ws.psdDelanoeEtAl14(t_min=0, t_max=400, n0star=-999.0, Dm=-999.0)
    ws.pndFromPsdBasic()


@arts_agenda
def pnd_agenda_AbelBoutle12(ws):
    ws.ExtractFromMetaSingleScatSpecies(
        ws.pnd_size_grid, ws.scat_meta, "diameter_max", ws.agenda_array_index
    )
    ws.Copy(ws.psd_size_grid, ws.pnd_size_grid)
    ws.ScatSpeciesSizeMassInfo(x_unit="dmax", species_index=ws.agenda_array_index)
    ws.psdAbelBoutle12(t_min=0, t_max=400)
    ws.pndFromPsdBasic()


# Field et al 2007
@arts_agenda
def pnd_agenda_FieldEtAl07ML(ws):
    ws.ExtractFromMetaSingleScatSpecies(
        ws.pnd_size_grid, ws.scat_meta, "diameter_max", ws.agenda_array_index
    )
    ws.Copy(ws.psd_size_grid, ws.pnd_size_grid)
    ws.ScatSpeciesSizeMassInfo(x_unit="dmax", species_index=ws.agenda_array_index)
    ws.psdFieldEtAl07(regime="ML", t_min=0, t_max=400)
    ws.pndFromPsdBasic()


@arts_agenda
def pnd_agenda_FieldEtAl07TR(ws):
    ws.ExtractFromMetaSingleScatSpecies(
        ws.pnd_size_grid, ws.scat_meta, "diameter_max", ws.agenda_array_index
    )
    ws.Copy(ws.psd_size_grid, ws.pnd_size_grid)
    ws.ScatSpeciesSizeMassInfo(x_unit="dmax", species_index=ws.agenda_array_index)
    ws.psdFieldEtAl07(regime="TR", t_min=0, t_max=400)
    ws.pndFromPsdBasic()


@arts_agenda
def pnd_agenda_McFarquaharHeymsfield97(ws):
    ws.ExtractFromMetaSingleScatSpecies(
        ws.pnd_size_grid, ws.scat_meta, "diameter_volume_equ", ws.agenda_array_index
    )
    ws.Copy(ws.psd_size_grid, ws.pnd_size_grid)
    ws.ScatSpeciesSizeMassInfo(x_unit="dveq", species_index=ws.agenda_array_index)
    ws.psdMcFarquaharHeymsfield97(t_min=0, t_max=400)
    ws.pndFromPsdBasic()


def set_additional_pnd_agendas(ws):
    ws.pnd_agenda_Delanoe14 = pnd_agenda_Delanoe14
    ws.pnd_agenda_AbelBoutle12 = pnd_agenda_AbelBoutle12
    ws.pnd_agenda_FieldEtAl07ML = pnd_agenda_FieldEtAl07ML
    ws.pnd_agenda_FieldEtAl07TR = pnd_agenda_FieldEtAl07TR
    ws.pnd_agenda_McFarquaharHeymsfield97 = pnd_agenda_McFarquaharHeymsfield97

    return ws


# =============================================================================


@arts_agenda
def iy_radar_agenda_singlepol(ws):
    ws.Ignore(ws.iy_id)
    ws.iy_transmitterSinglePol()
    ws.ppathStepByStep(cloudbox_on=0)
    ws.iyRadarSingleScat(trans_in_jacobian=1)
    ws.Touch(ws.geo_pos)


# =============================================================================
# aux functions
# =============================================================================


def create_agendas_in_ws(ws):

    pnd_agenda_list = []

    ws, pnd_agenda_SB06_list = create_pnd_agendas_SB06_in_WS(ws)
    ws, pnd_agenda_MY05_list = create_pnd_agendas_MY05_in_WS(ws)
    ws, pnd_agenda_CG_list = create_pnd_agendas_CG_in_WS(ws)
    ws, pnd_agenda_additional_list = create_additional_pnd_agendas_in_WS(ws)

    pnd_agenda_list = (
        pnd_agenda_SB06_list
        + pnd_agenda_MY05_list
        + pnd_agenda_CG_list
        + pnd_agenda_additional_list
    )

    pnd_agenda_list = [item.split("_agenda_")[1] for item in pnd_agenda_list]
    pnd_agenda_list.sort()

    return ws, pnd_agenda_list
