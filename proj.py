#!/usr/bin/env python
import logging, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.interpolate
import scipy.optimize
from ad import adnumber                 # for automatic differentation
from numpy import abs, arcsinh, log10, nan, pi, sqrt

# ----------------------------------------------------------------------------
# math
# ----------------------------------------------------------------------------

def cbrt(x):
    '''Cube root.  Needed to avoid getting complex numbers and also for
    compatibility with ad (which scipy.special.cbrt is not).'''
    if x >= 0:
        return x**(1./3.)
    return -(-x)**(1./3.)

def solve_equation(equation, **kwargs):
    x, _, ier, _ = scipy.optimize.fsolve(equation, full_output=True, **kwargs)
    return x if ier == 1 else nan       # return NaN if solution not found

def solve_equation_within(equation, xmin, xmax, attempts=10, **kwargs):
    for x0 in np.linspace(xmin, xmax, attempts):
        x = solve_equation(equation, x0=x0, **kwargs)
        if xmin <= x and x < xmax:
            return x
    return nan

# ----------------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------------

def create_figure(nx=1, ny=1):
    figure = plt.figure()
    axes = [figure.add_subplot(nx, ny, i + 1) for i in range(nx * ny)]
    return figure, axes

def plot_map(figure, axes, func, xmin, xmax, ymin, ymax,
             xnum=50, ynum=50, **kwargs):
    xs, ys = np.meshgrid(np.linspace(xmin, xmax, xnum),
                         np.linspace(ymin, ymax, ynum))
    zs = func(xs, ys)
    return axes.imshow(zs, origin="lower",
                       extent=(xmin, xmax, ymin, ymax), **kwargs)

def twinplot(x1, y1, x2, y2, fmt1="r-", fmt2="b-",
             color1="r", color2="b", xlabel="", ylabel1="", ylabel2="",
             ymin1=None, ymax1=None, ymin2=None, ymax2=None):
    '''Plot two curves on the same plot but with differing y-scales.'''
    fig, ax1 = plt.subplots()
    ax1.plot(x1, y1, fmt1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1, color=color1)
    ax1.set_ylim(ymin1, ymax1)
    for tl in ax1.get_yticklabels():
        tl.set_color(color1)
    ax2 = ax1.twinx()
    ax2.plot(x2, y2, fmt2)
    ax2.set_ylabel(ylabel2, color=color2)
    ax2.set_ylim(ymin2, ymax2)
    for tl in ax2.get_yticklabels():
        tl.set_color(color2)

# ----------------------------------------------------------------------------
# thermodynamics
# ----------------------------------------------------------------------------

def number_density_to_momentum(n):
    '''Compute the Fermi momentum from the number density for a Fermi gas.'''
    return cbrt(3. * pi**2 * n)

def momentum_to_number_density(k):
    '''Compute the number density from Fermi momentum for a Fermi gas.'''
    return k**3 / (3. * pi**2)

def ideal_fermi_gas_energy_density(k, m):
    '''Energy density of an ideal Fermi gas.'''
    mu = ideal_fermi_gas_chemical_potential(k, m)
    return (k * mu * (k**2 + mu**2) - m**4 * arcsinh(k / m)) / (8. * pi**2)

def ideal_fermi_gas_chemical_potential(k, m):
    '''Chemical potential of an ideal Fermi gas.'''
    return sqrt(m**2 + k**2)

def make_ideal_fermi_gas(m):
    @np.vectorize
    def epsilon(n):
        k = number_density_to_momentum(n)
        return ideal_fermi_gas_energy_density(k, m)
    @np.vectorize
    def mu(n):
        k = number_density_to_momentum(n)
        return ideal_fermi_gas_chemical_potential(k, m)
    return epsilon, mu

def nucleon_interaction_density(n_n, n_p):
    '''[V_np] Interaction density using mean-field Skyrme model.  Eq (28)'''
    k_n = number_density_to_momentum(n_n)
    k_p = number_density_to_momentum(n_p)
    n_B = n_n + n_p
    return (.5 * T_0 * ((.5 * X_0 + 1.) * n_B**2
                        - (X_0 + .5) * (n_p**2 + n_n**2))
            + .15  * (T_1 + T_2) * n_B * (n_n * k_n**2 + n_p * k_p**2)
            + .075 * (T_2 - T_1) * ((n_n * k_n)**2 + (n_p * k_p)**2)
            + .25  * T_3 * n_B * n_n * n_p)

def nonrelativistic_kinetic_density(n, m):
    '''Nonrelativistic kinetic energy density.  Eq (23)'''
    k = number_density_to_momentum(n)
    return .3 * k**2 * n / m

def nucleon_energy_density(n_n, n_p):
    '''Energy density for nucleons.  Eq (23)'''
    return (nonrelativistic_kinetic_density(n_n, M_N)
            + nonrelativistic_kinetic_density(n_p, M_P)
            + nucleon_interaction_density(n_n, n_p)
            + n_n * M_N + n_p * M_P)

@np.vectorize
def proton_chemical_potential(n_n, n_p):
    n_p = adnumber(n_p)
    return nucleon_energy_density(n_n, n_p).d(n_p)

@np.vectorize
def neutron_chemical_potential(n_n, n_p):
    n_n = adnumber(n_n)
    return nucleon_energy_density(n_n, n_p).d(n_n)

@np.vectorize
def total_energy_density_and_pressure(n_n):
    n_p = solve_for_proton_number_density(n_n)
    n_e = n_p
    epsilon_np = nucleon_energy_density(n_n, n_p)
    epsilon_e  = electron_energy_density(n_e)
    epsilon    = epsilon_np + epsilon_e
    mu_n = proton_chemical_potential(n_n, n_p)
    mu_p = neutron_chemical_potential(n_n, n_p)
    mu_e = electron_chemical_potential(n_e)
    return (epsilon, mu_n * n_n + mu_p * n_p + mu_e * n_e - epsilon)

def beta_equilibrium_equation(n_n, n_p):
    return (electron_chemical_potential(n_p)
            + proton_chemical_potential(n_n, n_p)
            - neutron_chemical_potential(n_n, n_p))

def solve_for_proton_number_density(n_n):
    N_P_MIN = N_EMPTY
    N_P_MAX = .12 * N_0
    N_N_MAX = 3.9 * N_0
    equation = lambda n_p: beta_equilibrium_equation(n_n, n_p)
    if n_n > N_N_MAX:
        return N_EMPTY
    n_p = solve_equation_within(equation, xmin=N_P_MIN, xmax=N_P_MAX)
    # if beta-equilibrium can't be achieved, then just use pure neutron matter
    if np.isnan(n_p):
        return N_EMPTY
    return n_p

def neutron_density_at_pressure(p, guess):
    '''Find the neutron number density at the given pressure.'''
    equation = lambda n_n: total_energy_density_and_pressure(n_n)[1] - p
    logging.info("solving for neutron number density at " +
                 "pressure = {0}/fm**3...".format(p))
    n_n = solve_equation(equation, x0=guess)[0]
    logging.info("  neutron number density = {0}/fm**3".format(n_n))
    return n_n

def neutron_density_at_zero_pressure():
    '''Find the positive neutron number density at zero pressure.'''
    N_N_AT_P_ZERO_GUESS = .42
    return neutron_density_at_pressure(0., N_N_AT_P_ZERO_GUESS)

def get_eos_table(n_n_begin, n_n_mid, n_n_end):
    try:
        return np.loadtxt("eos.txt").transpose() / HBAR_C
    except IOError:
        logging.info("regenerating EOS table...")
        pass
    n_n = np.concatenate([
        np.linspace(n_n_begin, n_n_mid, 50),
        np.logspace(log10(n_n_mid), log10(n_n_end), 200),
    ])
    epsilon, p = total_energy_density_and_pressure(n_n)
    epsilon_p = np.array([epsilon, p]).transpose()
    np.savetxt("eos.txt", epsilon_p * HBAR_C)
    return epsilon_p.transpose()

def interpolate_eos(epsilon, p):
    eos_raw = scipy.interpolate.interp1d(p, epsilon)
    @np.vectorize
    def eos(p):
        p = np.max([0., p]) / P_CONV
        try:
            epsilon = eos_raw(p) * P_CONV
            return epsilon
        except:
            logging.error("failed to evaluate EOS at " +
                          "pressure = {0} hbar c / fm^4".format(p))
            raise
    return eos

def plot_energy_density():
    '''Plot the energy density per nucleon for symmetric nuclear matter and
    pure neutron matter'''
    symmetric    = lambda n_B: nucleon_energy_density(n_B * .5, n_B * .5)
    pure_neutron = lambda n_B: nucleon_energy_density(n_B, 0)
    n_B = np.linspace(0, 2 * N_0, 250)
    plt.plot(n_B / N_0,    symmetric(n_B) / n_B * HBAR_C, "k")
    plt.plot(n_B / N_0, pure_neutron(n_B) / n_B * HBAR_C, "b")
    plt.xlabel("n / n_saturation")
    plt.ylabel("epsilon / n_B  /MeV")
    plt.show()

def plot_against_number_densities(figure, axes, func, n_n_max, n_p_max):
    '''Plot a function of (n_n, n_p) where number densities are in 1/fm**3.'''
    def f(n_n, n_p):
        return abs(func(n_n * N_0, n_p * N_0))
    img = plot_map(figure, axes, f,
                   xmin=N_EMPTY, xmax=n_n_max / N_0,
                   ymin=N_EMPTY, ymax=n_p_max / N_0,
                   xnum=100, ynum=100,
                   cmap="afmhot", interpolation="none", vmax=1)
    axes.set_xlabel("n_n /n_0")
    axes.set_ylabel("n_p /n_0")
    return img

def plot_beta_equilibrium_solutions(n_n_max, n_p_max):
    fig, [ax1, ax2] = create_figure(2, 1)
    def f(n_n, n_p):
        return neutron_chemical_potential(n_n, n_p) >= 0
    plot_against_number_densities(fig, ax1, f, n_n_max, n_p_max)
    img = plot_against_number_densities(fig, ax2, beta_equilibrium_equation,
                                        n_n_max, n_p_max)
    fig.colorbar(img)

def plot_proton_number_density_solutions():
    fig, [ax] = create_figure()
    n_ns = np.linspace(N_EMPTY, 4)
    n_ps = [solve_for_proton_number_density(N_0 * n_n) / N_0 for n_n in n_ns]
    ax.plot(n_ns, n_ps, "x")
    ax.set_xlabel("n_n")
    ax.set_ylabel("n_p")

def plot_eos(ax, epsilon, p, eos):
    p2 = np.logspace(-2, log10(np.max(p) - 1), 2000)
    ax.plot(p, epsilon, "x", label="data")
    ax.plot(p2, eos(p2 * P_CONV) / P_CONV, "r", label="interpolation")
    ax.set_xlabel("p /(hbar c / fm^4)")
    ax.set_ylabel("epsilon /(hbar c / fm^4)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()

# ----------------------------------------------------------------------------
# gravity
# ----------------------------------------------------------------------------
# this part uses solar mass units:
#   - pressure or energy density units: c^8 / (M_solar^2 G^3)
#   - mass units: M_solar
#   - length units: G M_solar / c^2

@np.vectorize
def simple_eos(p):
    '''A simple equation of state for noninteracting matter
    (from tov.ipynb by Michael Forbes).'''
    anr = 4.27675893
    ar  = 2.84265221
    p = np.max([p / P_CONV / 1e3 * HBAR_C, 1e-9])
    epsilon = (anr * p**(3./5.) + ar * p) * 1e3 / HBAR_C * P_CONV
    return epsilon

def tov_equation(mp, r, eos):
    '''The Tolman-Oppenheimer-Volkoff equation, which describes the hydrostatic
    equilibrium of a spherically symmetric object.'''
    [m, p] = mp
    epsilon = eos(p)
    deriv_m = 4. * pi * r**2 * epsilon
    deriv_p = (
        -m * epsilon / r**2
        * (1. + p / epsilon)
        * (1. + 4. * pi * r**3 * p / m)
        / (1. - 2. * m / r)
    )
    return [deriv_m, deriv_p]

def mass_pressure_profile(rs, p0, eos):
    '''Compute the distribution of mass and pressure in a spherically symmetric
    object.'''
    m0 = 4. * pi * rs[0]**3 * eos(p0) / 3.
    return scipy.integrate.odeint(
        tov_equation,
        [m0, p0],
        rs,
        args=(eos,),
    ).transpose()

def mass_radius(rs, p0, eos):
    ms, ps = mass_pressure_profile(rs, p0=p0, eos=eos)
    indices = np.where(ps <= P_EMPTY)[0]
    if not len(indices):
        logging.warn("can't find edge of star " +
                     "(central_pressure = {0} c^8 / (G^3 M_solar^2))".format(p0))
        return nan, nan
    edge = indices[0]
    return ms[edge], rs[edge]

def plot_mass_pressure_profile(r, m, p):
    twinplot(
        r * R_HSSOL, m,
        r * R_HSSOL, p,
        xlabel="radius /km",
        ylabel1="m(r) /M_solar",
        ylabel2="p(r) /(M_solar/km**3)",
        ymin1=0, ymin2=0,
    )

def plot_mass_radius(R, M):
    fig, [ax] = create_figure()
    ax.plot(R * R_HSSOL, M, "-x")
    ax.set_xlabel("R /km")
    ax.set_ylabel("M /M_solar")

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

HBAR_C = 197.326972    # [MeV*fm]
M_E  =    .51 / HBAR_C # [hbar/(fm*c)] (mass of an electron)
M_P  = 938.27 / HBAR_C # [hbar/(fm*c)] (mass of a proton)
M_N  = 939.56 / HBAR_C # [hbar/(fm*c)] (mass of a neutron)
M_MU = 105.66 / HBAR_C # [hbar/(fm*c)] (mass of a muon)
N_0  =    .16          # [1/fm**3]     (nuclear saturation density)

R_HSSOL = 1.47703573   # [km]          (half of solar Schwarzschild radius)
M_SOLAR = 5.654591e57  # [hbar/(fm*c)] (mass of the Sun)

N_EMPTY = 1e-10        # [1/fm**3] ("zero" number density to avoid div by zero)
P_EMPTY = 1e-9

# conversion factor from femtometer units to solar mass units
P_CONV = 1e54 * R_HSSOL**3 / M_SOLAR

# convert GeV/fm^3 to solar mass units
GEV_FM3 = P_CONV * 1e3 / HBAR_C

# constants for the Skyrme interaction
T_0 = -5.93  # [fm**2]
T_1 =  2.97  # [fm**4]
T_2 =  -.137 # [fm**4]
T_3 = 47.29  # [fm**5]
X_0 =   .34  # [dimensionless]

electron_energy_density, electron_chemical_potential = \
    make_ideal_fermi_gas(M_E)
muon_energy_density, muon_chemical_potential = \
    make_ideal_fermi_gas(M_MU)

# ----------------------------------------------------------------------------

def main():
    logging.basicConfig(
        format="[%(levelname)s] %(message)s",
        level=logging.INFO,
    )

    # --- units: femtometers ---

    #plot_beta_equilibrium_solutions(4*N_0, .5*N_0)

    #plot_proton_number_density_solutions()

    epsilon, p = get_eos_table(
        n_n_begin=neutron_density_at_zero_pressure() - 1e-10,
        n_n_mid=4 * N_0,
        n_n_end=neutron_density_at_pressure(1e10, 1e3),
    )

    eos = interpolate_eos(epsilon, p)

    fig, [ax1, ax2] = create_figure(1, 2)
    plot_eos(ax1, epsilon, p, eos)
    plot_eos(ax2, epsilon, p, eos)
    ax1.set_xlim(right=4)
    ax1.set_ylim(top=6)

    # --- units: solar mass ---

    r = np.linspace(
        1e-4, # km
        40,   # km
        1000,
    ) / R_HSSOL

    p0 = 0.2 * GEV_FM3
    m, p = mass_pressure_profile(r, p0, eos)
    plot_mass_pressure_profile(r, m, p)

    p0s = np.logspace(
        -3, # GeV/fm^3
        3,  # GeV/fm^3
    ) * GEV_FM3
    M, R = np.array([mass_radius(r, p0, eos) for p0 in p0s]).transpose()

    plot_mass_radius(R, M)

    plt.show()

if __name__ == "__main__":
    main()
