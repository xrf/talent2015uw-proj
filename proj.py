#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.integrate as spi
import scipy.optimize as spo
from ad            import adnumber
from numpy         import arcsinh, sqrt, nan, pi

def cbrt(x):
    '''Cube root.  Needed to avoid getting complex numbers and also for
    compatibility with ad (which scipy.special.cbrt is not).'''
    if x >= 0:
        return x**(1./3.)
    return -(-x)**(1./3.)

def pressure(e_pn, e_e, mu_n, mu_p, mu_e, n_n, n_p, n_e):
    return -e_pn - e_e + mu_n * n_n + mu_p * n_p + mu_e * n_e

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
    n_B = np.linspace(0, 2 * .16, 250)
    plt.plot(n_B / .16,    symmetric(n_B) / n_B * MEV_FM, "k")
    plt.plot(n_B / .16, pure_neutron(n_B) / n_B * MEV_FM, "b")
    plt.xlabel("n / n_saturation")
    plt.ylabel("epsilon / n_B  /MeV")
    plt.show()

def vectorize(f):
    return np.vectorize(f, [float])

@np.vectorize
def proton_chemical_potential(n_n, n_p):
    n_p = adnumber(n_p)
    return nucleon_energy_density(n_n, n_p).d(n_p)

@np.vectorize
def neutron_chemical_potential(n_n, n_p):
    n_n = adnumber(n_n)
    return nucleon_energy_density(n_n, n_p).d(n_n)

def fermi_gas_chemical_potential(n, m):
    '''Chemical potential of a noninteracting, relativistic Fermi gas.'''
    k = number_density_to_momentum(n)
    return sqrt(m**2 + k**2)

@np.vectorize
def electron_chemical_potential(n_e):
    return fermi_gas_chemical_potential(n_e, m=M_E)

@np.vectorize
def muon_chemical_potential(n_e):
    return fermi_gas_chemical_potential(n_e, m=M_MU)

def solve_equation(equation, x0):
    x, _, ier, _ = spo.fsolve(equation, x0=x0, full_output=True)
    return x if ier == 1 else nan       # return NaN if solution not found

def beta_equilibrium_equation(n_n, n_p):
    return (electron_chemical_potential(n_p)
            + proton_chemical_potential(n_n, n_p)
            - neutron_chemical_potential(n_n, n_p))

def solve_for_neutron_number_density(n_p):
    equation = lambda n_n: beta_equilibrium_equation(n_n, n_p)
    n_n = np.linspace(1e-10, 10 * .16, 250)##
    plt.plot(n_n, equation(n_n))##
    return solve_equation(equation, x0=1.)

def solve_for_proton_number_density(n_n):
    equation = lambda n_p: beta_equilibrium_equation(n_n, n_p)
    n_p = np.linspace(1e-5, 100 * .16, 250)##
    plt.plot(n_p, [equation(n_p_) for n_p_ in n_p])##
    return solve_equation(equation, x0=1.)

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

plt.gca().axhline(y=0, color="gray")

# note: it seems that only one of the roots is physical (nonnegative num
# density) perhaps we should solve n_e and n_p first (resp.) and figure out
# where they cross the x-axis?
nvar = np.linspace(1e-5, 4, 100) * .16
n_n = nvar
n_p = .07 * .16
plt.xlabel("n_n")
plt.plot(n_p, electron_chemical_potential(n_p), label="e")
plt.plot(nvar/.16, proton_chemical_potential(n_n, n_p), label="p")
plt.plot(nvar/.16, neutron_chemical_potential(n_n, n_p), label="n")
plt.plot(nvar/.16, beta_equilibrium_equation(n_n, n_p), label="beta")
plt.legend()
plt.show()
exit()

#n_ps = np.linspace(4, 11, 10)
#n_ns = [solve_for_neutron_number_density(.16 * n_p) for n_p in n_ps]
#plt.plot(n_ps, n_ns)
n_ns = np.linspace(1, 21, 10)
n_ps = [solve_for_proton_number_density(.16 * n_n) for n_n in n_ns]
# plt.plot(n_ps, n_ns)
plt.show()

exit()

def energy_density_gas(mu, mass):
    b = sqrt(mu**2 - mass**2)
    epsilon = (b * mu * (2. * b**2 + mass**2)
               - mass**4 * arcsinh(b / mass)) / 8.
    return epsilon / pi**2

def number_density(mu, mass):
    return sqrt(mu**2 - mass**2)**3 / (3 * pi**2)

def chemical_potential(mass, momentum):
    return sqrt(mass * mass + momentum * momentum)

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
n_0 = 0.16

plt.plot(df["mu_n"] * MEV_FM, df["n_p"] / (df["n_p"] + np.nan_to_num(df["n_n"])))
# against baryon density (?)
#plt.plot(df["n_p"] + df["n_n"], df["n_p"] / (df["n_p"] + np.nan_to_num(df["n_n"])))
plt.ylim(0, 1.1)
plt.ylabel("n_p / n_baryon")

#plt.plot(df["mu_n"] * MEV_FM, df["n_p"])
#plt.plot(df["mu_n"] * MEV_FM, np.nan_to_num(df["n_n"]))
plt.xlabel("mu_p  /MeV")
plt.show()
