# -*- coding: utf-8 -*-
"""This module contains classes handling different convection schemes."""
import abc

import numpy as np
import typhon
from scipy.interpolate import interp1d

from konrad import constants
from konrad.component import Component
from konrad.surface import SurfaceFixedTemperature


__all__ = [
    'Convection',
    'NonConvective',
    'HardAdjustment',
    'RelaxedAdjustment',
]


def energy_difference(T_2, T_1, sst_2, sst_1, dp, eff_Cp_s):
    """
    Calculate the energy difference between two atmospheric profiles (2 - 1).

    Parameters:
        T_2: atmospheric temperature profile (2)
        T_1: atmospheric temperature profile (1)
        sst_2: surface temperature (2)
        sst_1: surface temperature (1)
        dp: pressure thicknesses of levels,
            must be the same for both atmospheric profiles
        eff_Cp_s: effective heat capacity of surface
    """
    Cp = constants.isobaric_mass_heat_capacity
    g = constants.g

    dT = T_2 - T_1  # convective temperature change of atmosphere
    dT_s = sst_2 - sst_1  # of surface

    termdiff = - np.sum(Cp/g * dT * dp) + eff_Cp_s * dT_s

    return termdiff


class Convection(Component, metaclass=abc.ABCMeta):
    """Base class to define abstract methods for convection schemes."""
    @abc.abstractmethod
    def stabilize(self, atmosphere, lapse, surface, timestep):
        """Stabilize the temperature profile by redistributing energy.

        Parameters:
              atmosphere (konrad.atmosphere.Atmosphere): Atmosphere model.
              lapse (ndarray): Temperature lapse rate [K/day].
              surface (konrad.surface): Surface model.
              timestep (float): Timestep width [day].
        """


class NonConvective(Convection):
    """Do not apply convection."""
    def stabilize(self, *args, **kwargs):
        pass


class HardAdjustment(Convection):
    """Instantaneous adjustment of temperature profiles"""
    def stabilize(self, atmosphere, lapse, surface, timestep):

        # Find convectively adjusted temperature profile.
        T_new, T_s_new = self.convective_adjustment(
            p=atmosphere['plev'],
            phlev=atmosphere['phlev'],
            T_rad=atmosphere['T'][0, :],
            lapse=lapse,
            surface=surface,
            timestep=timestep,
        )
        # Update atmospheric temperatures as well as surface temperature.
        atmosphere['T'][0, :] = T_new
        surface.temperature.values[0] = T_s_new

    def convective_adjustment(self, p, phlev, T_rad, lapse, surface,
                              timestep=0.1):
        """
        Find the energy-conserving temperature profile using a iterative
        procedure with test profiles. Update the atmospheric temperature
        profile to this one.

        Parameters:
            p (ndarray): pressure levels
            phlev (ndarray): half pressure levels
            T_rad (ndarray): old atmospheric temperature profile
            lapse (konrad.lapserate): lapse rate in K/km
            surface (konrad.surface):
                surface associated with old temperature profile
            timestep (float): only required for slow convection
        """
        # The threshold is scaled with the effective heat capacity of the
        # surface. Otherwise very thick surfaces may never reach the target.
        near_zero = float(surface.heat_capacity / 1e13)

        # Interpolate density and lapse rate on pressure half-levels.
        density1 = typhon.physics.density(p, T_rad)
        density = interp1d(p, density1, fill_value='extrapolate')(phlev[:-1])

        g = constants.earth_standard_gravity
        lp = -lapse / (g * density)

        # find energy difference if there is no change to surface temp due to
        # convective adjustment. in this case the new profile should be
        # associated with an increase in energy in the atmosphere.
        surfaceTpos = surface.temperature.data
        T_con, diffpos = self.test_profile(T_rad, p, phlev, surface,
                                           surfaceTpos, lp,
                                           timestep=timestep)

        # this is the temperature profile required if we have a set-up with a
        # fixed surface temperature, then the energy does not matter.
        if isinstance(surface, SurfaceFixedTemperature):
            return T_con, surface.temperature
        # for other cases, if we find a decrease or approx no change in energy,
        # the atmosphere is not being warmed by the convection,
        # as it is not unstable to convection, so no adjustment is applied
        if diffpos < near_zero:
            return T_con, surface.temperature

        # if the atmosphere is unstable to convection, a fixed surface temp
        # produces an increase in energy (as convection warms the atmosphere).
        # this surface temperature is an upper bound to the energy-conserving
        # surface temperature.
        # taking the surface temperature as the coldest temperature in the
        # radiative profile gives us a lower bound.
        surfaceTneg = np.array([np.min(T_rad)])
        eff_Cp_s = surface.heat_capacity
        diffneg = eff_Cp_s * (surfaceTneg - surface['temperature'])
        # good guess for energy-conserving profile (unlikely!)
        if np.abs(diffneg) < near_zero:
            return T_con, surfaceTneg

        # NOTE (lkluft): Dirty workaround to always initialize `surfaceT`.
        # I encountered situations where the while-loop did not run a single
        # iteration and therefore the return-statement failed.
        surfaceT = (surfaceTneg + (surfaceTpos - surfaceTneg)
                    * (-diffneg) / (-diffneg + diffpos))

        # Now we have a upper and lower bound for the surface temperature of
        # the energy conserving profile. Iterate to get closer to the energy-
        # conserving temperature profile.
        counter = 0
        while diffpos >= near_zero and -diffneg >= near_zero:
            surfaceT = (surfaceTneg + (surfaceTpos - surfaceTneg)
                        * (-diffneg) / (-diffneg + diffpos))
            T_con, diff = self.test_profile(T_rad, p, phlev, surface, surfaceT,
                                            lp, timestep=timestep)
            if diff > 0:
                diffpos = diff
                surfaceTpos = surfaceT
            else:
                diffneg = diff
                surfaceTneg = surfaceT

            # to avoid getting stuck in a loop if something weird is going on
            counter += 1
            if counter == 100:
                raise ValueError(
                    "No energy conserving convective profile can be found"
                )

        # save new temperature profile
        return T_con, surfaceT

    def test_profile(self, T_rad, p, phlev, surface, surfaceT, lp,
                     timestep=0.1):
        """
        Assuming a particular surface temperature (surfaceT), create a new
        profile, following the specified lapse rate (lp) for the region where
        the convectively adjusted atmosphere is warmer than the radiative one.

        Parameters:
            T_rad (ndarray): old atmospheric temperature profile
            p (ndarray): pressure levels
            phlev (ndarray): half pressure levels
            surface (konrad.surface):
                surface associated with old temperature profile
            surfaceT (float): surface temperature of the new profile
            lp (ndarray): lapse rate in K/Pa
            timestep (float): not required in this case

        Returns:
            ndarray: new atmospheric temperature profile
            float: energy difference between the new profile and the old one
        """
        # dp, thicknesses of atmosphere layers, for energy calculation
        dp = np.diff(phlev)
        # for lapse rate integral
        dp_lapse = np.hstack((np.array([p[0] - phlev[0]]), np.diff(p)))
        T_con = surfaceT - np.cumsum(dp_lapse * lp)
        if np.any(T_con > T_rad):
            contop = np.max(np.where(T_con > T_rad))
            T_con[contop+1:] = T_rad[contop+1:]
        else:
            T_con = T_rad
            contop = 0

        # If run with a fixed surface temperature, always return the
        # convective profile starting from the current surface temperature.
        if isinstance(surface, SurfaceFixedTemperature):
            return T_con, 0.

        eff_Cp_s = surface.heat_capacity

<<<<<<< HEAD
        diff = energy_difference(T_con, T_rad, surfaceT, surface.temperature,
                                 dp, eff_Cp_s)
=======
        diff = energy_difference(T_con, T_rad, surfaceT,
                                 surface['temperature'][-1], dp, eff_Cp_s)
>>>>>>> 5344382... Fix convection

        return T_con, float(diff)


class RelaxedAdjustment(HardAdjustment):
    """Adjustment with relaxed convection in upper atmosphere.

    This convection scheme allows for a transition regime between a
    convectively driven troposphere and the radiatively balanced stratosphere.
    """
    def __init__(self, tau=None):
        """
        Parameters:
            tau (ndarray): Array of convective timescale values [days]
        """
        self.convective_tau = tau

    def get_convective_tau(self, p):
        """Return a convective timescale profile.

        Parameters:
            p (ndarray): Pressure levels [Pa].

        Returns:
            ndarray: Convective timescale profile [days].
        """
        if self.convective_tau is not None:
            return self.convective_tau

        tau0 = 1/24 # 1 hour
        tau = tau0*np.exp(p[0] / p)

        return tau

    def test_profile(self, T_rad, p, phlev, surface, surfaceT, lp,
                     timestep=0.1):
        """
        Assuming a particular surface temperature (surfaceT), create a new
        profile, using the convective timescale and specified lapse rate (lp).

        Parameters:
            T_rad (ndarray): old atmospheric temperature profile
            p (ndarray): pressure levels
            phlev (ndarray): half pressure levels
            surface (konrad.surface):
                surface associated with old temperature profile
            surfaceT (float): surface temperature of the new profile
            lp (ndarray): lapse rate in K/Pa
            timestep (float): not required in this case

        Returns:
            ndarray: new atmospheric temperature profile
            float: energy difference between the new profile and the old one
        """
        dp = np.diff(phlev)
        dp_lapse = np.hstack((np.array([p[0] - phlev[0]]), np.diff(p)))

        tau = self.get_convective_tau(p)

        tf = 1 - np.exp(-timestep / tau)
        T_con = T_rad * (1 - tf) + tf * (surfaceT - np.cumsum(dp_lapse * lp))

        # If run with a fixed surface temperature, always return the
        # convective profile starting from the current surface temperature.
        if isinstance(surface, SurfaceFixedTemperature):
            return T_con, 0.

        eff_Cp_s = surface.heat_capacity

        diff = energy_difference(T_con, T_rad, surfaceT, surface.temperature,
                                 dp, eff_Cp_s)
        return T_con, float(diff)


