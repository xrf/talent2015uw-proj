#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
import scipy.optimize as spo
from ad import adnumber                 # for automatic differentation
from numpy import abs, arcsinh, sqrt, min, max, nan, pi

def cbrt(x):
    '''Cube root.  Needed to avoid getting complex numbers and also for
    compatibility with ad (which scipy.special.cbrt is not).'''
    if x >= 0:
        return x**(1./3.)
    return -(-x)**(1./3.)

def total_pressure(n_n):
    n_e = n_p
    n_p = solve_for_proton_number_density(n_n)
    epsilon_np = nucleon_energy_density(n_n, n_p)
    epsilon_e  = electron_energy_density(n_e)
    mu_n = proton_chemical_potential(n_n, n_p)
    mu_p = neutron_chemical_potential(n_n, n_p)
    mu_e = electron_chemical_potential(n_e)
    return mu_n * n_n + mu_p * n_p + mu_e * n_e - epsilon_np - epsilon_e

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

def number_density_to_momentum(n):
    '''Compute the Fermi momentum from the number density for a Fermi gas.'''
    return cbrt(n * (3. * pi**2))

def momentum_to_number_density(k):
    '''Compute the number density from Fermi momentum for a Fermi gas.'''
    return k**3 / (3. * pi**2)

def nucleon_energy_density(n_n, n_p):
    '''Energy density for nucleons.  Eq (23)'''
    return (nucleon_kinetic_density(n_n, n_p)
            + nucleon_interaction_density(n_n, n_p))

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

@np.vectorize
def proton_chemical_potential(n_n, n_p):
    n_p = adnumber(n_p)
    return nucleon_energy_density(n_n, n_p).d(n_p)

@np.vectorize
def neutron_chemical_potential(n_n, n_p):
    n_n = adnumber(n_n)
    return nucleon_energy_density(n_n, n_p).d(n_n)

def ideal_fermi_gas_energy_density(k, m):
    '''Energy density of an ideal Fermi gas.'''
    mu = fermi_gas_chemical_potential(k, m)
    return (k * mu * (k**2 + mu**2) - m**4 * arcsinh(k / m)) / (8. * pi**2)

def ideal_fermi_gas_chemical_potential(k, m):
    '''Chemical potential of an ideal Fermi gas.'''
    return sqrt(m**2 + k**2)

def make_ideal_fermi_gas(m):
    def epsilon(n):
        k = number_density_to_momentum(n)
        return fermi_gas_energy_density(k, m)
    def mu(n):
        k = number_density_to_momentum(n)
        return fermi_gas_chemical_potential(k, m)
    return epsilon, mu

electron_energy_density, electron_chemical_potential \
    = make_ideal_fermi_gas(M_E)
muon_energy_density, muon_chemical_potential \
    = make_ideal_fermi_gas(M_MU)

def solve_equation(equation, **kwargs):
    x, _, ier, _ = spo.fsolve(equation, full_output=True, **kwargs)
    return x if ier == 1 else nan       # return NaN if solution not found

def beta_equilibrium_equation(n_n, n_p):
    return (electron_chemical_potential(n_p)
            + proton_chemical_potential(n_n, n_p)
            - neutron_chemical_potential(n_n, n_p))

def solve_for_neutron_number_density(n_p):
    equation = lambda n_n: beta_equilibrium_equation(n_n, n_p)
    # n_n = np.linspace(1e-10, 10 * N_0, 250)##
    # plt.plot(n_n, equation(n_n))##
    return solve_equation(equation, x0=1.)

def solve_for_proton_number_density(n_n):
    equation = lambda n_p: beta_equilibrium_equation(n_n, n_p)
    # n_p = np.linspace(1e-5, 100 * N_0, 250)##
    # plt.plot(n_p, [equation(n_p_) for n_p_ in n_p])##
    return solve_equation(equation, x0=1)

# note: energy-like quantities are in units of 1/fm, equivalent to 197.33 MeV
MEV_FM = 197.33

M_E  =    .51 / MEV_FM # 1/fm    (mass of an electron)
M_P  = 938.27 / MEV_FM # 1/fm    (mass of a proton)
M_N  = 939.56 / MEV_FM # 1/fm    (mass of a neutron)
M_MU = 105.66 / MEV_FM # 1/fm    (mass of a muon)
N_0  =    .16          # 1/fm**3 (nuclear saturation density)

# constants for the Skyrme interaction
T_0 = -5.93  # fm**2
T_1 =  2.97  # fm**4
T_2 =  -.137 # fm**4
T_3 = 47.29  # fm**5
X_0 =   .34  # (dimensionless)

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

def plot_against_number_densities(figure, axes, func):
    '''Plot a function of (n_n, n_p) where number densities are in 1/fm**3.'''
    def f(n_n, n_p):
        return abs(func(n_n * N_0, n_p * N_0))
    img = plot_map(figure, axes, f,
                   xmin=1e-5, xmax=6, ymin=1e-5, ymax=6, xnum=100, ynum=100,
                   cmap="afmhot", interpolation="none", vmax=1)
    axes.set_xlabel("n_n /n_0")
    axes.set_ylabel("n_p /n_0")
    return img

def plot_beta_equilibrium_solutions():
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    def f(n_n, n_p):
        return np.logical_and(proton_chemical_potential(n_n, n_p) >= 0,
                              neutron_chemical_potential(n_n, n_p) >= 0)
    plot_against_number_densities(fig, ax1, f)
    img = plot_against_number_densities(fig, ax2, beta_equilibrium_equation)
    fig.colorbar(img)

fig, [ax] = create_figure(1, 1)
n_ns = np.linspace(1e-5, 6)
n_ps = [solve_for_proton_number_density(N_0 * n_n) / N_0 for n_n in n_ns]
ax.plot(n_ns, n_ps, "x")
ax.set_xlabel("n_n")
ax.set_ylabel("n_p")
plt.show()
exit()

def equation_of_state(fermi_momentum):
    '''Calculate the equation of state of an electrically neutral Fermi gas of
    electrons, neutrons, and protons in beta-equilibrium with the given Fermi
    momentum of electrons (which is the same for protons since n_e = n_p).'''
    mu_e = chemical_potential(M_E, fermi_momentum)
    mu_p = chemical_potentialenergy(M_P, fermi_momentum)
    mu_n = mu_e + mu_p
    P = pressure(mu_n, M_N) + pressure(mu_p, M_P) + pressure(mu_e, M_E)
    epsilon = (energy_density(mu_n, M_N) +
               energy_density(mu_p, M_P) +
               energy_density(mu_e, M_E))
    n_e = number_density(mu_e, M_E)
    n_n = number_density(mu_n, M_N)
    n_p = number_density(mu_p, M_P)
    return {
        "mu_p": mu_p,
        "mu_n": mu_n,
        "n_e": n_e,
        "n_n": n_n,
        "n_p": n_p,
        "epsilon": epsilon,
        "P": P,
    }

def solve_k_F(P):
    '''Find the Fermi momentum that matches the given pressure'''
    return spo.fsolve(lambda k_F: equation_of_state(k_F)["P"] - P, x0=0.8)

def equation_of_state_by_pressure(pressure):
    '''Calculate the equation of state by total pressure.'''
    return equation_of_state(solve_k_F(pressure))

kF1 = 0.01
k_F_range = np.concatenate([np.linspace(0, kF1, 400), np.linspace(kF1, 100, 100)])
df = pd.DataFrame.from_records(equation_of_state(k_F) for k_F in k_F_range)
#n = n_n + n_p

plt.plot(df["mu_n"] * MEV_FM, df["n_p"] / (df["n_p"] + np.nan_to_num(df["n_n"])))
# against baryon density (?)
#plt.plot(df["n_p"] + df["n_n"], df["n_p"] / (df["n_p"] + np.nan_to_num(df["n_n"])))
plt.ylim(0, 1.1)
plt.ylabel("n_p / n_baryon")

#plt.plot(df["mu_n"] * MEV_FM, df["n_p"])
#plt.plot(df["mu_n"] * MEV_FM, np.nan_to_num(df["n_n"]))
plt.xlabel("mu_p  /MeV")
plt.show()
