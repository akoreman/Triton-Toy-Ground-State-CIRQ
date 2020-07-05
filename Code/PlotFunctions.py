# -*- coding: utf-8 -*-
"""
Applied Quantum Algorithms Mini-Project 1, Tim Koreman (2418541), Leiden University (2020).

Function to plot the three body contact density.
"""
import numpy as np
import cirq as c
import matplotlib.pyplot as plt

import StatePreparation as SP
import TimeEvolution as TE

class MethodError(Exception):        
    def __string__(self):
        return "Invalid time evolution method."

def PlotThreeBodyContact(u, v, t, theta, phi, t_start, t_end, t_sweep_size, n_measurements, n_trotter, method, preparation_method, n_ancilla = 0):
    '''
    Funtion to plot the three body contact density.

    Parameters
    ----------
    u : float
        One the parameters in the Hamiltonian, see report for description.
    v : float
        One the parameters in the Hamiltonian, see report for description.
    t : float
        One the parameters in the Hamiltonian, see report for description.
    theta : float
        Angle to initialize the ansatz with.
    phi : float
        Angle to initialize the ansatz with.
    t_start : int
        Time point where the sweep starts.
    t_end : int
        Time point where the sweep ends.
    t_sweep_size : int
        Number of steps plotted between the start and end time.
    n_measurements : int
        How many times the simulator runs the system to calculate the expectation values with.
    n_trotter : int
        Number of trotter steps used in the time evoltution.
    method : string
        Choose the method used to calculate the time evoltuion. Options are: 'gates', 'exact' and 'trotter exact', denoting timeevolution by a circuit, full matrix exponential and trotterized matrix exponential respectively.
    preparation_method: string
        Choose the method used to initialize the approximated ground state. Options are: 'ansatz' and 'qpe'. Make sure a non zero value for n_ancilla is chosen when using qpe.
    n_ancilla: int, optional
        Defaults to zero, only used when preparation_method = 'qpe', choose non zero value when using qpe.

    Raises
    ------
    MethodError
        When a method string is chosen that doesn't match one of the options an error is raised.

    Returns
    -------
    None.

    '''
    results_array = []

    t_sweep = np.linspace(t_start, t_end, t_sweep_size)
      
    for i in range(0, t_sweep_size):
        qubits = [c.LineQubit(0), c.LineQubit(1), c.LineQubit(2), c.LineQubit(3)]
        
        circ = c.Circuit()
        
        if preparation_method == 'ansatz':
            circ.append(SP.StatePreparation(theta,phi)(*qubits))
        elif preparation_method == 'qpe':
             circ.append(SP.QPEStatePreparation(theta, phi, qubits, n_ancilla, u, v, t, 1.0))
        else:
            raise MethodError("Invalid time evolution method.")
                
        for j in range(0, n_trotter):
            if method == 'gates':
                circ.append(TE.TrotterizedEvolutionGate(u,v,t,t_sweep[i]/n_trotter)(*qubits))
            elif method == 'exact':
                circ.append(TE.ExactEvolution(u,v,t,t_sweep[i]/n_trotter)(*qubits)) 
            elif method == 'trotterexact':
                circ.append(TE.ExactEvolutionTrotterOne(u,v,t,t_sweep[i]/n_trotter)(*qubits))
            else:
                raise MethodError("Invalid time evolution method.")
                
        circ.append(c.measure(*qubits, key='Zmeasurements'))
          
        simulator = c.Simulator()
        result = simulator.run(circ, repetitions=n_measurements)
        
        if preparation_method == 'ansatz':
            results_array.append(result.histogram(key='Zmeasurements')[0]/n_measurements)
        elif preparation_method == 'qpe':
            ancilla_histogram = result.histogram(key='ancilla')
            ancilla_values = result.data["ancilla"].values 
            indices = np.array(np.where(ancilla_values == 788))[0]
        
            filtered_results = result.data["Zmeasurements"].values[indices]
            binned_results = np.bincount(filtered_results)
            results_array.append(binned_results[0]/len(filtered_results))
        else:
            raise MethodError("Invalid time evolution method.")
    
    plt.plot(t_sweep, results_array, '.')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$C_3 (t)$')
    
    plt.savefig('ThreeBodyContact.pdf')
    
    plt.show()

