import numpy as np
import matplotlib.pyplot as plt
from ad    import adnumber
from numpy import arcsinh, sqrt, pi

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
    return (n * (3. * pi**2))**(1./3.)

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

def proton_chemical_potential(n_n, n_p):
    n_p = adnumber(n_p)
    return nucleon_energy_density(n_n, n_p).d(n_p)

def neutron_chemical_potential(n_n, n_p):
    n_n = adnumber(n_n)
    return nucleon_energy_density(n_n, n_p).d(n_n)

# note: energy-like quantities are in units of 1/fm, equivalent to 197.33 MeV
MEV_FM = 197.33

M_E =    .51 / MEV_FM # 1/fm    (mass of an electron)
M_P = 938.27 / MEV_FM # 1/fm    (mass of a proton)
M_N = 939.56 / MEV_FM # 1/fm    (mass of a neutron)
N_0 =    .16          # 1/fm**3 (nuclear saturation density)

# constants for the Skyrme interaction
T_0 = -5.93  # fm**2
T_1 =  2.97  # fm**4
T_2 =  -.137 # fm**4
T_3 = 47.29  # fm**5
X_0 =   .34  # (dimensionless)
