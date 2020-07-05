# -*- coding: utf-8 -*-
"""
Applied Quantum Algorithms Mini-Project 1, Tim Koreman (2418541), Leiden University (2020).

Functions used to benchmark our implementation of the VQE with the built in optimizer in openfermion.
"""

import openfermioncirq as ofc

import openfermion as of
import cirq as c
import sympy
import numpy as np
from openfermioncirq.optimization import COBYLA, OptimizationParams

class Ansatz(ofc.VariationalAnsatz):  
    '''
    Ansatz as defined in the report as used by openfermion to optimize the VQE. Inherits openfermioncirq.VariationalAnsatz.
    '''
    def params(self):
        return [self.theta,self.phi]

    def operations(self, qubits):
        self.theta = sympy.Symbol('theta')
        self.phi = sympy.Symbol('phi')
        
        theta = self.theta
        phi = self.phi
        
        yield [c.ry(theta)(qubits[i]) for i in range(0,4)]
    
        yield c.CZ(qubits[0],qubits[3])
        yield c.CZ(qubits[1],qubits[2])
        
        yield [c.ry(phi)(qubits[i]) for i in range(2,4)]
        
        yield c.CZ(qubits[0],qubits[3])
        yield c.CZ(qubits[1],qubits[2])
            
    def _generate_qubits(self):
        return [c.LineQubit(i) for i in range(0,4)]
    
      
def Hamiltonian(u,v,t):
    '''
    Function that returns the hamiltonian with the given parameters by using openfermion.QubitOperators.

    Parameters
    ----------
    u : float
        One the parameters in the Hamiltonian, see report for description.
    v : float
        One the parameters in the Hamiltonian, see report for description.
    t : float
        One the parameters in the Hamiltonian, see report for description.

    Returns
    -------
    hamiltonian : Sum of openfermion.QubitOperators
        The hamiltonian as defined in the report with the parameters as provided.

    '''
    hamiltonian =  -2 * t *       of.QubitOperator('X0') 
    hamiltonian += -2 * t *       of.QubitOperator('X1') 
    hamiltonian += -2 * t *       of.QubitOperator('X2') 
    hamiltonian += -2 * t *       of.QubitOperator('X3') 
    hamiltonian += v/16   *       of.QubitOperator('Z0 Z3') 
    hamiltonian += v/16   *       of.QubitOperator('Z1 Z2') 
    hamiltonian += v/16   *       of.QubitOperator('Z0 Z1 Z2') 
    hamiltonian += v/16   *       of.QubitOperator('Z0 Z2 Z3') 
    hamiltonian += v/16   *       of.QubitOperator('Z0 Z1 Z3') 
    hamiltonian += v/16   *       of.QubitOperator('Z1 Z2 Z3') 
    hamiltonian += (v + 4*u)/16 * of.QubitOperator('Z0 Z1') 
    hamiltonian += (v + 4*u)/16 * of.QubitOperator('Z2 Z3') 
    hamiltonian += (v + 4*u)/16 * of.QubitOperator('Z0 Z2')  
    hamiltonian += (v + 4*u)/16 * of.QubitOperator('Z1 Z3') 
    hamiltonian += (v + 4*u)/16 * of.QubitOperator('Z0')  
    hamiltonian += (v + 4*u)/16 * of.QubitOperator('Z1') 
    hamiltonian += (v + 4*u)/16 * of.QubitOperator('Z2') 
    hamiltonian += (v + 4*u)/16 * of.QubitOperator('Z3') 
    hamiltonian += (v + 4*u)/16 * of.QubitOperator('Z0 Z1 Z2 Z3') 
    
    return hamiltonian
    
def OptimizeVQEOpenFermion(u,v,t):
    '''
    Function to optimize a VQE for the hamiltonian and ansatz we are looking at by using the built in optimizer in openfermion.

    Parameters
    ----------
    U : float
        One the parameters in the Hamiltonian, see report for description.
    V : float
        One the parameters in the Hamiltonian, see report for description.
    t : float
        One the parameters in the Hamiltonian, see report for description.

    Returns
    -------
    List
        List containing 4 values: 1) theta at minimum, 2) phi at minimum, 3) energy at minimum, 4) Error in the energy w.r.t. the exact ground state energy.

    '''
    ansatz = Ansatz()
    hamiltonian = Hamiltonian(u,v,t)
    
    objective = ofc.HamiltonianObjective(hamiltonian)
    
    study = ofc.VariationalStudy(
        name='TritonToy',
        ansatz=ansatz,
        objective=objective)
        
    optimization_params = OptimizationParams(
        algorithm=COBYLA,
        initial_guess=[1,1])
    result = study.optimize(optimization_params)
        
    minimum_energy = result.optimal_value + (8*t + 3*u/4 + v/16)
    minimum_energy_error = result.optimal_value - np.amin(of.eigenspectrum(hamiltonian))
    
    minimum_theta = result.optimal_parameters[0]
    minimum_phi = result.optimal_parameters[1]
    
    text = "Minimum E = {}, at theta = {} and phi = {} error = {}.".format(minimum_energy, minimum_theta, minimum_phi, minimum_energy_error)   
    print(text)
    
    return [minimum_theta, minimum_phi, minimum_energy, minimum_energy_error]
