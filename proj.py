#!/usr/bin/env python
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.interpolate
import scipy.optimize
from ad import adnumber                 # for automatic differentation
from numpy import abs, arcsinh, log, min, max, nan, pi, sqrt

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

def solve_equation_within(equation, xmin, xmax, attempts, **kwargs):
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
# physics
# ----------------------------------------------------------------------------

def number_density_to_momentum(n):
    '''Compute the Fermi momentum from the number density for a Fermi gas.'''
    return cbrt(n * (3. * pi**2))

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
    '''[V_np] Interaction using mean-field Skyrme model.  Eq (28)'''
    k_n = number_density_to_momentum(n_n)
    k_p = number_density_to_momentum(n_p)
    n_B = n_n + n_p
    return (.5 * T_0 * ((.5 * X_0 + 1.) * n_B * n_B
                        - (X_0 + .5) * (n_p**2 + n_n**2))
            + .15  * (T_1 + T_2) * n_B * (n_n * k_n**2 + n_p * k_p**2)
            + .075 * (T_2 - T_1) * ((n_n * k_n)**2 + (n_p * k_p)**2)
            + .25  * T_3 * n_B * n_n * n_p)

def nucleon_kinetic_density(n_n, n_p):
    '''[T_np] Nuclear kinetic energy density.  Eq (23)'''
    k_n = number_density_to_momentum(n_n)
    k_p = number_density_to_momentum(n_p)
    return .3 * (k_n**2 * n_n / M_N + k_p**2 * n_p / M_P)

def nucleon_energy_density(n_n, n_p):
    '''Energy density for nucleons.  Eq (23)'''
    return (nucleon_kinetic_density(n_n, n_p)
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
    N_P_ATTEMPTS = 10
    N_N_MAX = 3.9 * N_0
    equation = lambda n_p: beta_equilibrium_equation(n_n, n_p)
    if n_n > N_N_MAX:
        return N_EMPTY
    n_p = solve_equation_within(equation, xmin=N_P_MIN, xmax=N_P_MAX,
                                attempts=N_P_ATTEMPTS)
    # if beta-equilibrium can't be achieved, then just use pure neutron matter
    if np.isnan(n_p):
        return N_EMPTY
    return n_p

def plot_energy_density():
    '''Plot the energy density per nucleon for symmetric nuclear matter and
    pure neutron matter'''
    symmetric    = lambda n_B: nucleon_energy_density(n_B * .5, n_B * .5)
    pure_neutron = lambda n_B: nucleon_energy_density(n_B, 0)
    n_B = np.linspace(0, 2 * N_0, 250)
    plt.plot(n_B / N_0,    symmetric(n_B) / n_B * MEV_FM, "k")
    plt.plot(n_B / N_0, pure_neutron(n_B) / n_B * MEV_FM, "b")
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

def get_eos_table(n_n):
    try:
        return np.loadtxt("eos.txt").transpose() / MEV_FM
    except IOError:
        print("regenerating EOS table...")
        pass
    epsilon, p = total_energy_density_and_pressure(n_n)
    epsilon_p = np.array([epsilon, p]).transpose()
    np.savetxt("eos.txt", np.concatenate([[0, 0], epsilon_p]) * MEV_FM)
    return epsilon_p.transpose()

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

# note: energy-like quantities are in units of 1/fm, equivalent to 197.33 MeV
MEV_FM = 197.326972                     # hbar c /(MeV fm)

G = 2.61210003e-40     # gravitational constant /(fm^2 c^3/hbar)
M_E  =    .51 / MEV_FM # 1/fm    (mass of an electron)
M_P  = 938.27 / MEV_FM # 1/fm    (mass of a proton)
M_N  = 939.56 / MEV_FM # 1/fm    (mass of a neutron)
M_MU = 105.66 / MEV_FM # 1/fm    (mass of a muon)
N_0  =    .16          # 1/fm**3 (nuclear saturation density)
A_Z  = 2               # A/Z ratio

N_EMPTY = 1e-10               # "zero" number density (to avoid singularities)

# constants for the Skyrme interaction
T_0 = -5.93  # fm**2
T_1 =  2.97  # fm**4
T_2 =  -.137 # fm**4
T_3 = 47.29  # fm**5
X_0 =   .34  # (dimensionless)

electron_energy_density, electron_chemical_potential \
    = make_ideal_fermi_gas(M_E)
muon_energy_density, muon_chemical_potential \
    = make_ideal_fermi_gas(M_MU)

# ----------------------------------------------------------------------------

def hydrostatic_equation(mp, r, eos):
    [m, p] = mp
    epsilon = eos(p)
    print(r, epsilon, m)
    deriv_m = 4. * pi * r**2 * epsilon
    deriv_p = (
        -G * m * epsilon / r**2
        * (1. + p / epsilon)
        * (1. + 4. * pi * r**3 * p / m)
        / (1. - 2. * G * m / r)
    )
    return [deriv_m, deriv_p]

def mass_pressure_profile(rs, m0, p0, eos):
    return scipy.integrate.odeint(
        hydrostatic_equation,
        [m0, p0],
        rs,
        args=(eos,),
    ).transpose()

def white_dwarf_mass_radius(rs, m0, p0, eos):
    ms, ps = mass_pressure_profile(rs, m0=m0, p0=p0, eos=eos)
    indices = np.where(ps <= 0)[0]
    if not len(indices):
        return nan, nan
    edge = indices[0]
#    print("LL",edge, np.where(ps <= 0))
    print( edge, ms[edge], rs[edge])
    return ms[edge], rs[edge]

def main():

    #plot_beta_equilibrium_solutions(4*N_0, .5*N_0)

    # fig, [ax] = create_figure()
    # n_ns = np.linspace(N_EMPTY, 4)
    # n_ps = [solve_for_proton_number_density(N_0 * n_n) / N_0 for n_n in n_ns]
    # ax.plot(n_ns, n_ps, "x")
    # ax.set_xlabel("n_n")
    # ax.set_ylabel("n_p")

    N_MID = 4 * N_0
    N_END = 10 * N_0
    n_n = np.concatenate([
        np.linspace(N_EMPTY, N_MID, 100),
        np.linspace(N_MID, N_END, 50),
    ])
    epsilon, p = get_eos_table(n_n)

    # fig, [ax] = create_figure()
    # ax.plot(epsilon, p, "x")
    # ax.set_xlabel("epsilon")
    # ax.set_ylabel("p")

    p0 = 1e-8
    m0 = 1e-8
    r_min = 1e-10 * 1e18
    r_max = 2e22
    rs = np.linspace(r_min, r_max, 100)

    eos_raw = scipy.interpolate.interp1d(p, epsilon)
    def eos(p):
        if p < 0:
            return 1e-99
        return eos_raw(p)
    print(np.min(p), np.max(p))

    ms, ps = mass_pressure_profile(rs, m0, p0, eos)
    twinplot(rs, ms, rs, ps, xlabel="radius", ylabel1="m", ylabel2="p",
             ymin1=0, ymin2=0)

    plt.show()

    # p0s = np.linspace(1, 1e10 / MEV_FM, 100)
    # ms, rs = zip(*(white_dwarf_mass_radius(rs, p0, eos) for p0 in p0s))
    # for m, r in zip(ms, rs):
    #     print(m, r)

    # plt.plot(rs, ms, "x")

if "run" in sys.argv[1:]:
    main()
    exit()
