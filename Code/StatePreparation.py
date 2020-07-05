# -*- coding: utf-8 -*-
"""
Applied Quantum Algorithms Mini-Project 1, Tim Koreman (2418541), Leiden University (2020).

Functions used to prepare the state as used for the rest of the simulations.
"""
import numpy as np
import cirq as c
import random

import HamiltonianDefinitions as HD
import StatePreparation as SP
import TimeEvolution as TE
import ExactEigenvectors as EE

def EnergyFunction(u, v, t, theta, phi, n_measurements):
    '''
    Function to measure the expectation value for our ansatz and hamiltonian at a given set of angles and parameters.

    Parameters
    ----------
    u : float
        One the parameters in the Hamiltonian, see report for description.
    v : float
        One the parameters in the Hamiltonian, see report for description.
    t : float
        One the parameters in the Hamiltonian, see report for description.
    theta : float
        One of the two angles of the ansatz.
    phi : float
        One of the two angles of the ansatz.
    n_measurements : int
        How many times the simulator runs the system to calculate the expectation values with. Estimation improves with increasing N but so does runtime.

    Returns
    -------
    Float
        Estimate of the energy at the provided angles and parameters.

    '''
    C1 = (v + 4*u)/16
    C2 = v/16
    
    h = [HD.H1, HD.H2, HD.H3, HD.H4, HD.H5, HD.H6, HD.H7, HD.H8, HD.H9, HD.H10, HD.H11, HD.H12, HD.H13, HD.H14, HD.H15, HD.H16, HD.H17, HD.H18, HD.H19]
    h_coeffiecients = [-2 * t, -2 * t, -2 * t, -2 * t, C2, C2, C2, C2, C2, C2, C1, C1, C1, C1, C1, C1, C1, C1, C1, 8*t + 3 * u / 4 + v / 16] #Coeff list is 1 longer to account for constant added term.
      
    energy = 0
    
    for i in range(0,len(h)):
        qubits = [c.LineQubit(0), c.LineQubit(1), c.LineQubit(2), c.LineQubit(3)]
        ancilla = c.LineQubit(4)
        
        circ = c.Circuit()
        circ.append(SP.StatePreparation(theta, phi)(*qubits))        
        circ.append(h[i](qubits, ancilla))
                                
        simulator = c.Simulator()
        result = simulator.run(circ, repetitions=n_measurements)
                    
        expectation_value = (result.histogram(key='Zmeasurements')[0] - result.histogram(key='Zmeasurements')[1])/n_measurements
        
        energy += expectation_value * h_coeffiecients[i]
    
    return energy + h_coeffiecients[len(h)]

def OptimizeVQEGridSearch(u, v, t, n_measurements, sweep_size):    
    '''
    Function to optimize the VQE using a naieve grid search. Not used for results in report.
    
    Parameters
    ----------
    u : float
        One the parameters in the Hamiltonian, see report for description.
    v : float
        One the parameters in the Hamiltonian, see report for description.
    t : float
        One the parameters in the Hamiltonian, see report for description.
    n_measurements : int
        How many times the simulator runs the system to calculate the expectation values with.
    sweep_size : int
        How many steps the gridsearch should use, more steps reach higher accuracy.

    Returns
    -------
    list
        List containing 4 values: 1) theta at minimum, 2) phi at minimum, 3) energy at minimum, 4) Error in the energy w.r.t. the exact ground state energy.

    '''
    theta_sweep = np.linspace(0, 2 * np.pi, sweep_size)
    phi_sweep = np.linspace(0, 2 * np.pi, sweep_size)
    
    minimum_energy = 10000
    minimum_theta = 0
    minimum_phi = 0
    
    for i in range(0, sweep_size):
        for j in range(0, sweep_size):
            energy = EnergyFunction(u, v, t, theta_sweep[i], phi_sweep[j], n_measurements)
            
            print("i = {} j = {} Emin = {}".format(i,j, minimum_energy))
            
            if (energy < minimum_energy): 
                minimum_energy = energy
                minimum_theta = theta_sweep[i]
                minimum_phi = phi_sweep[j]
       
    minimum_energy_error = minimum_energy - EE.ExactEigenVectors(u, v, t)[0][0]
    
    text = "Minimum E = {}, at theta = {} and phi = {} error = {}.".format(minimum_energy, minimum_theta, minimum_phi, minimum_energy_error)       
    print(text) 
        
    return [minimum_theta, minimum_phi, minimum_energy, minimum_energy_error]


def OptimizeVQENelderMead(u, v, t, n_measurements, cutoff_variance, max_iterations = 100, rho = 1, chi = 2, gamma = 1/2, sigma = 1/2): 
    '''
    Function to optimize the VQE using a Nalder-Mead search.

    Parameters
    ----------
    u : float
        One the parameters in the Hamiltonian, see report for description.
    v : float
        One the parameters in the Hamiltonian, see report for description.
    t : float
        One the parameters in the Hamiltonian, see report for description.
    n_measurements : int
        How many times the simulator runs the system to calculate the expectation values with.
    cutoff_variance: float
        The algorithm stops when the energy values of the simplex reach this variance.
    max_iterations: int
        If the cut off is not satisfied after max_iterations start again with a fresh simplex.
    rho : float, optional
        One of the parameters in the Nalder-Mead search. The default is 1.
    chi : float, optional
        One of the parameters in the Nalder-Mead search. The default is 2.
    gamma : float, optional
        One of the parameters in the Nalder-Mead search. The default is 1/2.
    sigma : float, optional
        One of the parameters in the Nalder-Mead search. The default is 1/2.

    Returns
    -------
    list
        List containing 4 values: 1) theta at minimum, 2) phi at minimum, 3) energy at minimum, 4) Error in the energy w.r.t. the exact ground state energy.

    '''
    # initialization of the simplex at random values and determining the energy at those values of the parameters.
    simplex_theta = np.array([1.0, 0.0, 0.0])
    simplex_phi = np.array([1.0, 0.0, 0.0])
    simplex_energy = np.array([0.0, 0.0, 0.0])
        
    simplex_theta[1] = random.uniform(0, 2*np.pi)
    simplex_theta[2] = random.uniform(0, 2*np.pi)
    
    simplex_phi[1] = random.uniform(0, 2*np.pi)
    simplex_phi[2] = random.uniform(0, 2*np.pi)
       
    simplex_energy[0] = EnergyFunction(u, v, t, simplex_theta[0], simplex_phi[0], n_measurements)
    simplex_energy[1] = EnergyFunction(u, v, t, simplex_theta[1], simplex_phi[1], n_measurements)
    simplex_energy[2] = EnergyFunction(u, v, t, simplex_theta[2], simplex_phi[2], n_measurements)
      
    n_iterations = 0
    
    while True:
        # ordering step.
        order = np.argsort(simplex_energy) 
       
        simplex_phi = simplex_phi[order]
        simplex_theta = simplex_theta[order]
        simplex_energy = simplex_energy[order]
            
        # calculate centroid of the 2 points of the simplex with the lowest energy.
        centroid_theta = np.mean(simplex_theta[:2])
        centroid_phi = np.mean(simplex_phi[:2])
          
        
        # calculate the reflected point.
        reflected_theta = (centroid_theta + rho*(centroid_theta - simplex_theta[2]))%(2*np.pi)
        reflected_phi = (centroid_phi + rho*(centroid_phi - simplex_phi[2]))%(2*np.pi)
        
        reflected_energy = EnergyFunction(u, v, t, reflected_theta, reflected_phi, n_measurements)
         
        if (reflected_energy < simplex_energy[1] and simplex_energy[0] <= reflected_energy):
            simplex_theta[2] = reflected_theta
            simplex_phi[2] = reflected_phi
            simplex_energy[2] = reflected_energy            
        elif (reflected_energy < simplex_energy[0]):
            expanded_theta = (centroid_theta + chi*(reflected_theta - centroid_theta))%(2*np.pi)
            expanded_phi = (centroid_phi + chi*(reflected_phi - centroid_phi))%(2*np.pi)
            
            expanded_energy = EnergyFunction(u, v, t, expanded_theta, expanded_phi, n_measurements)
            
            if (expanded_energy < reflected_energy):
                simplex_theta[2] = expanded_theta
                simplex_phi[2] = expanded_phi
                simplex_energy[2] = expanded_energy
            else:
                simplex_theta[2] = reflected_theta
                simplex_phi[2] = reflected_phi
                simplex_energy[2] = reflected_energy
        else:
            contracted_theta = (centroid_theta + gamma * (simplex_theta[2] - centroid_theta))%(2*np.pi)
            contracted_phi = (centroid_phi + gamma * (simplex_phi[2] - centroid_phi))%(2*np.pi)
            
            contracted_energy = EnergyFunction(u, v, t, contracted_theta, contracted_phi, n_measurements)
            
            if (contracted_energy < simplex_energy[2]):
                simplex_theta[2] = contracted_theta
                simplex_phi[2] = contracted_phi
                simplex_energy[2] = contracted_energy
            else:
                simplex_theta[1] = (simplex_theta[0] + sigma * (simplex_theta[1] - simplex_theta[0]))%(2*np.pi)
                simplex_phi[1] = (simplex_phi[0] + sigma * (simplex_phi[1] - simplex_phi[0]))%(2*np.pi)
                simplex_energy[1] = EnergyFunction(u, v, t, simplex_theta[1], simplex_phi[1], n_measurements)
                
                simplex_theta[2] = (simplex_theta[0] + sigma * (simplex_theta[2] - simplex_theta[0]))%(2*np.pi)
                simplex_phi[2] = (simplex_phi[0] + sigma * (simplex_phi[2] - simplex_phi[0]))%(2*np.pi)
                simplex_energy[2] = EnergyFunction(u, v, t, simplex_theta[2], simplex_phi[2], n_measurements)
            
        n_iterations += 1
        
        if(np.var(simplex_energy) < cutoff_variance): break
        
        # If cut off is not satisfied after max_iterations start again.
        if(n_iterations > max_iterations):       
            simplex_theta = np.array([0.0, 0.0, 0.0])
            simplex_phi = np.array([0.0, 0.0, 0.0])
            simplex_energy = np.array([0.0, 0.0, 0.0])
                             
            simplex_theta[0] = random.uniform(0, 2*np.pi)
            simplex_theta[1] = random.uniform(0, 2*np.pi)
            simplex_theta[2] = random.uniform(0, 2*np.pi)
            
            simplex_phi[0] = random.uniform(0, 2*np.pi)
            simplex_phi[1] = random.uniform(0, 2*np.pi)
            simplex_phi[2] = random.uniform(0, 2*np.pi)
                       
            simplex_energy[0] = EnergyFunction(u, v, t, simplex_theta[0], simplex_phi[0], n_measurements)
            simplex_energy[1] = EnergyFunction(u, v, t, simplex_theta[1], simplex_phi[1], n_measurements)
            simplex_energy[2] = EnergyFunction(u, v, t, simplex_theta[2], simplex_phi[2], n_measurements)
            n_iterations = 0
                       
    minimum_theta = simplex_theta[0]
    minimum_phi = simplex_phi[0]
    minimum_energy = simplex_energy[0]
    
    #calculate error w.r.t. exact eigenvalue found using numpy.
    minimum_energy_error = minimum_energy - EE.ExactEigenVectors(u, v, t)[0][0]

    text = "Minimum E = {}, at theta = {} and phi = {} error = {}.".format(minimum_energy, minimum_theta, minimum_phi, minimum_energy_error)   
    print(text)  
        
    return [minimum_theta, minimum_phi, minimum_energy, minimum_energy_error]

class StatePreparation(c.Gate):
    '''
    Gate that implements the ansatz for a given value of theta/phi. Inherits c.Gate.
    '''
    def _num_qubits_(self):
        return 4
            
    def __init__(self, theta, phi):
        self.theta = theta
        self.phi = phi
        
    def _decompose_(self, qubits):    
        yield [c.ry(self.theta)(qubits[i]) for i in range(0,4)]
        
        yield c.CZ(qubits[0],qubits[3])
        yield c.CZ(qubits[1],qubits[2])
        
        yield [c.ry(self.phi)(qubits[i]) for i in range(2,4)]
        
        yield c.CZ(qubits[0],qubits[3])
        yield c.CZ(qubits[1],qubits[2])

def QPEStatePreparation(theta, phi, qubits, n_ancilla,u, v, t, delta_t):
    '''
    Function that implements the QPE to for state preparation.

    Parameters
    ----------
    theta : float
        Angle to initialize the ansatz with.
    phi : float
        Angle to initialize the ansatz with.
    qubits : c.LineQubit
        The collection of qubits the algorithm acts on.
    ancilla_qubits : int
        The umber of ancilla's the algorithm uses.
    u : float
        One the parameters in the Hamiltonian, see report for description.
    v : float
        One the parameters in the Hamiltonian, see report for description.
    t : float
        One the parameters in the Hamiltonian, see report for description.
    delta_t : float
        The time period used in the unitary.

    Yields
    ------
    c.Gates
        Yield multiple gates as implemented in cirq.

    '''
    ancilla_qubits = [c.LineQubit(i) for i in range(4, 4 + n_ancilla)]
    
    yield StatePreparation(theta, phi)(*qubits)
       
    for i in range(0, n_ancilla ):
        yield c.H(ancilla_qubits[i])
        
    for i in range(0, n_ancilla ):  
        yield TE.TrotterizedEvolutionGate(u, v, t, delta_t * 2**i).controlled()(ancilla_qubits[i], *qubits)
            
    yield c.QFT(*ancilla_qubits, inverse = True)       
    yield c.measure(*ancilla_qubits, key='ancilla') 

