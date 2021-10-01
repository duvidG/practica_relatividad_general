#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 12:19:15 2021

@author: dgiron
"""

from imports_gen_rel import *
from ecuaciones_apuntes import biseccion

M_real = 2e30 # Solar mass (kg)
G = 6.67e-11 # Gravitational constant (SI)
c = 3e8 # Speed of light (m/s)

def straight_line(r_e, r_p, eta, M):
    """
    Calculates Delta T, i.e, the time delay due to gravitational effects neglecting the
    curvature of the light ray. Note: distance units can be given in any unit.

    Parameters
    ----------
    r_e : np.ndarray
        array with the distances from the Sun to the Earth for multiple MJDs.
    r_p : np.ndarray
        array with the distances from the Sun to the planet (Venus) for the same MJDs as r_e.
    eta : np.ndarray
        minimum distance from the light ray to the Sun.
    M : float
        mass of the Sun, in kg.

    Returns
    -------
    float
        time delay due to gravitational effects, in seconds.

    """
    return 4 * G * M / c ** 3 * np.log((r_p + np.sqrt(r_p ** 2 - eta ** 2)) * (r_e + np.sqrt(r_e ** 2 - eta ** 2)) / eta ** 2)

def shapiro(r_e, r_p, R, M):
    """
    Calculates Delta T, i.e, the time delay due to gravitational effects using
    the formula given in the Shapiro paper. Note: distance units can be given in any unit.

    Parameters
    ----------
    r_e : np.ndarray
        array with the distances from the Sun to the Earth for multiple MJDs.
    r_p : np.ndarray
        array with the distances from the Sun to the planet (Venus) for the same MJDs as r_e.
    R : np.ndarray
        distance from the planet (Venus) to the Earth.
    M : float
        mass of the Sun, in kg.

    Returns
    -------
    float
        time delay due to gravitational effects, in seconds.

    """
    return 4 * G * M / c ** 3 * np.log((r_e + r_p + R)/(r_e + r_p - R))


def main():
    """
    Instructions: given the two sets of data, from Arecibo and Haystack, a comparison between these data
    and the model specified in the Shapiro's paper is done. In order to draw the plot, line 248 should be uncommented.
    To estimate the mass of the Sun using the Chi square, line 251 should be uncommented.
    Note: the data files should be in the same directory as the main program, as well
    as the "imports.py" and "ecuaciones_apuntes.py" files, given in the github folder

    """
    # Data import
    arecibo = np.genfromtxt('arecibo.dat', delimiter=',')
    haystack = np.genfromtxt('haystack.dat', delimiter=',')
    geo = np.genfromtxt('geom.dat', delimiter='')

    # Save each column in an independent vector to facilitate the understanding
        # Arecibo
    mjd_are = arecibo[:, 0]
    delay_are = arecibo[:, 1]
    sigma_are_up = arecibo[:, 2]
    sigma_are_down = arecibo[:, 3]
    y_err_are = [sigma_are_up, sigma_are_down]

        # Haystack
    mjd_hay = haystack[:, 0]
    delay_hay = haystack[:, 1]
    sigma_hay_up = haystack[:, 2]
    sigma_hay_down = haystack[:, 3]
    y_err_hay = [sigma_hay_up, sigma_hay_down]

        # Geometry
    mjd_geo = geo[:, 0]
    r_p = geo[:, 1]
    r_e = geo[:, 2]
    eta = geo[:, 3]

    # Calculation of the distance between Earth and Venus (R) with eta, r_e and r_p (from Pitagoras theorem)
    R = np.sqrt(-1 * eta ** 2 + r_e ** 2) + np.sqrt(-1 * eta ** 2 + r_p ** 2)

    # Values for the delay using both models
    y_straight = straight_line(r_e, r_p, eta, M_real) * 1e6
    y_shapiro = shapiro(r_e, r_p, R, M_real) * 1e6

    def draw_plot():
        """
        Plots the Arecibo and Haystack data, comparing it to the straight line model obtained in class
        and the one given in Shapiro's paper. Both Arecibo and Haystack values are given with their respective
        errorbars.

        Returns
        -------
        None.

        """
        plt.clf()
        plt.errorbar(mjd_are, delay_are, yerr=y_err_are, fmt='.', label='Arecibo', fillstyle='none')
        plt.errorbar(mjd_hay, delay_hay, yerr=y_err_hay, fmt='.', label='Haystack')
        plt.plot(mjd_geo, y_straight, 'r-', label='Modelo prop. rectilínea', alpha=0.5)
        plt.plot(mjd_geo, y_shapiro, 'b--', label='Modelo Shapiro', alpha=0.5)
        plt.legend()
        plt.grid()
        plt.xlabel('Dias desde la conjuncion superior')
        plt.ylabel(r'$\Delta T$/$\mu$s')
        plt.savefig('delay.png', dpi=720)


    def chi_sq(M):
        """
        Obtains the value of chi square for a given mass of the Sun. For the calculation,
        it uses both the data from Arecibo and Haystack, taking into account their respective errors.

        Parameters
        ----------
        M : float
            mass of the Sun, in kg.

        Returns
        -------
        float
            normalized chi square.

        """
        y_shapiro = shapiro(r_e, r_p, R, M) * 1e6 # change to microseconds
        predicted = []
        delay_tot = np.concatenate((delay_are, delay_hay))
        mjd_tot = np.concatenate((mjd_are, mjd_hay))

        # In order to obtain the predicted values for a given MJD, the loop
        # looks for the closest day in the values given in the model (i.e. the minimum
        # of the difference between a certaint MJD of the observations and the ones
        # of the model). When this MJD is found, with its index, the value of the
        # delay is taken from the model delays. This is repeated for all the days
        # and the predicted delays are saved in a list.

        for x, y in zip(delay_tot, mjd_tot):
            minim = np.argmin(np.abs(y - mjd_geo))
            predicted.append(y_shapiro[minim])

        predicted = np.array(predicted)
        observed = np.concatenate((delay_are, delay_hay))
        sigma_up_tot = np.concatenate((sigma_are_up, sigma_hay_up))
        sigma_down_tot = np.concatenate((sigma_are_down, sigma_hay_down))

        # Chi square formula
        chi = np.sum(((observed - predicted) ** 2) / (((sigma_up_tot + sigma_down_tot) / 2) ** 2))
        return chi

    def derivative(x, h=1e25, f=chi_sq):
        """
        Calculates the derivative of a given function in a point using the derivative definition, a little
        bit processed in order to minimise the error.

        Parameters
        ----------
        x : float
            point where the derivative is going to be evaluated.
        h : float, optional
            increment (see derivative definition). The default is 1e25.
        f : function, optional
            function to be derived. The default is chi_sq.

        Returns
        -------
        float
            result of the derivative.

        """
        return (f(x + h)  - f(x - h)) / (2 * h)


    def estimate_solar_mass(min_plot, max_plot, bisection_precision, num_masses, plot):
        """
        Estimates the mass of the Sun using chi square. An algorithm based on the bisection equation solving method
        is used (see "ecuaciones_apuntes.py")

        Parameters
        ----------
        min_plot : float
            fraction of the mean value of the mass to use in the bisection algorithm. Should be less than one
        max_plot : float
            fraction of the mean value of the mass to use in the bisection algorithm. Should be more than one.
        bisection_precision : float
            minimum precision of the bisection algorithm result. Caution: if this value is bigger than the
            values of the y axis, the algorithm will just return the mean of min_plot and max_plot. For this case,
            is recommended a value of 1e-36
        num_masses : float
            number of elements of the masses array used in the plot.
        plot : bool
            condition to display the plot of chi square for multiple masses.

        Returns
        -------
        solar_mass : ufloat
            tuple with the estimation of the solar mass and its error.
            
        """
        # mass the minimizes chi square
        mass_min_chi = biseccion(derivative, min_plot * M_real, max_plot * M_real, bisection_precision)

        # Chi for that mass
        min_chi = chi_sq(mass_min_chi)

        # Array with masses to plot
        mm = np.linspace(min_plot * M_real, max_plot * M_real, num_masses)
        chi_plot = [chi_sq(i) for i in mm]
        if plot:
            plt.clf()
            plt.plot(mm, chi_plot)
            plt.plot([min(mm), max(mm)], [min_chi + 1, min_chi + 1], '--k', label=r'Maxima $\chi^2$ permitida')
            plt.plot(mass_min_chi, min_chi, 'rx', label=r'$\chi^2$ minima')
            plt.xlabel('Mass/kg')
            plt.ylabel(r'$\chi^2$')
            plt.legend()
            plt.savefig('chi2.png', dpi=720)

        verify_chi_condition = [j for i, j in zip(chi_plot, mm) if i <= min_chi  + 1]

        solar_mass = ufloat(np.mean(verify_chi_condition), np.std(verify_chi_condition))
        return solar_mass


    print('Chi_sq reduced with the real mass = {:.2f}' .format(chi_sq(M_real)/(len(delay_are)+len(delay_hay))))

    # Uncomment the following line to display the plot:
    #draw_plot()

    # Uncomment the following line to estimate the mass of the Sun using chi square
    print('Solar mass estimation: ', estimate_solar_mass(min_plot=0.95, max_plot=1.05, bisection_precision=1e-36, num_masses=1000, plot=True))

main()