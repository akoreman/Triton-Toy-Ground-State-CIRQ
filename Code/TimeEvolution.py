# -*- coding: utf-8 -*-
"""
Applied Quantum Algorithms Mini-Project 1, Tim Koreman (2418541), Leiden University (2020).

Functions and gates used for time evolution.
"""
import cirq as c
import numpy as np

from scipy.linalg import expm

import ExactEigenvectors as EE  

def TwoZGate(qubit1, qubit2, angle):
    '''
    Gate that implements a Z rotation by angle on qubit1 and qubit2
    '''
    yield c.CNOT(qubit1, qubit2)
    yield c.rz(angle)(qubit2)
    yield c.CNOT(qubit1, qubit2)
    
def ThreeZGate(qubit1, qubit2, qubit3, angle):
    '''
    Gate that implements a Z rotation by angle on qubit1, qubit2 and qubit3.
    '''
    yield c.CNOT(qubit1, qubit2)
    yield c.CNOT(qubit2, qubit3)
    yield c.rz(angle)(qubit3)
    yield c.CNOT(qubit2, qubit3)
    yield c.CNOT(qubit1, qubit2)
    
def FourZGate(qubit1, qubit2, qubit3, qubit4, angle):
    '''
    Gate that implements a Z rotation by angle on qubit1, qubit2, qubit3 and qubit4.
    '''
    yield c.CNOT(qubit1, qubit2)
    yield c.CNOT(qubit2, qubit3)
    yield c.CNOT(qubit3, qubit4)
    yield c.rz(angle)(qubit3)
    yield c.CNOT(qubit3, qubit4)
    yield c.CNOT(qubit2, qubit3)
    yield c.CNOT(qubit1, qubit2)
            
class TrotterizedEvolutionGate(c.Gate): 
    '''
    Gate that implemts one of the trotterized timesteps. Initializes with hamiltonian parameters u,v,t and angles phi,theta. Inherits c.Gate
    '''
    def _num_qubits_(self):
        return 4
    
    def _decompose_(self, qubits): 
        angle1 = self.angle1
        angle2 = self.angle2
        angle3 = self.angle3
               
        if (angle3==0):                  
            yield c.CNOT(qubits[0], qubits[2])
            yield c.CNOT(qubits[0], qubits[3])
            yield c.CNOT(qubits[1], qubits[2])
            
            yield c.rz(angle2)(qubits[2])
            yield c.rz(angle2)(qubits[3])
            
            yield c.CNOT(qubits[0], qubits[2])
            yield c.CNOT(qubits[1], qubits[3])
            
            yield c.rz(angle2)(qubits[2])
            yield c.rz(angle2)(qubits[3])
            
            yield c.CNOT(qubits[2], qubits[3])
            
            yield c.rz(angle2)(qubits[3])
            
            yield c.CNOT(qubits[0], qubits[3])
            yield c.CNOT(qubits[1], qubits[3])
            
            yield c.rz(angle2)(qubits[3])
            
            yield c.CNOT(qubits[2], qubits[3])
            yield c.CNOT(qubits[1], qubits[2])
            
            yield [c.rx(angle1)(qubits[i]) for i in range(0,4)]
        else:               
            yield TwoZGate(qubits[0], qubits[1], angle3)        
            yield TwoZGate(qubits[2], qubits[3], angle3)        
            yield TwoZGate(qubits[0], qubits[2], angle3)        
            yield TwoZGate(qubits[1], qubits[3],angle3)
            
            yield TwoZGate(qubits[0], qubits[3], angle2)       
            yield TwoZGate(qubits[1], qubits[2], angle2)
        
            yield ThreeZGate(qubits[0], qubits[1], qubits[2], angle2)      
            yield ThreeZGate(qubits[0], qubits[1], qubits[3], angle2)
            yield ThreeZGate(qubits[1], qubits[2], qubits[3], angle2)        
            yield ThreeZGate(qubits[0], qubits[2], qubits[3], angle2)
                  
            yield FourZGate(qubits[0], qubits[1], qubits[2], qubits[3], angle3)
            
            yield [c.rz(angle3)(qubits[i]) for i in range(0,4)]
            yield [c.rx(angle1)(qubits[i]) for i in range(0,4)]
    
    def __init__(self, u, v, t, delta_t):     
        self.angle1 = 1* 4*t*delta_t
        self.angle2 = -1* delta_t * v/8
        self.angle3 = -1*  delta_t * (v + 4*u)/8
        
def ExactTimeEvolution(u,v,t,delta_t):
    '''
    Function to use numpy to calculate the 'exact' matrix exponential to calculate the e^(- i H t) unitary time evolution operator.

    Parameters
    ----------
    u : float
        One the parameters in the Hamiltonian, see report for description.
    V : float
        One the parameters in the Hamiltonian, see report for description.
    t : float
        One the parameters in the Hamiltonian, see report for description.
    delta_t : float
        The amount of time the operator evolves the system by.

    Returns
    -------
    numpy array (2 dim)
        numpy array describing the matrix exponential.

    '''
    # definition of the 3 pauli matrices as np.array.
    X = np.array([[0,1],[1, 0]])
    Z = np.array([[1,0],[0,-1]])
    I = np.array([[1,0],[0, 1]])
    
    h0 = (8*t + 3 * u / 4 + v / 16) * EE.FourKronecker(I, I, I, I)
    
    h1 = -2*t * EE.FourKronecker(X,I,I,I)
    h2 = -2*t * EE.FourKronecker(I,X,I,I)
    h3 = -2*t * EE.FourKronecker(I,I,X,I)
    h4 = -2*t * EE.FourKronecker(I,I,I,X)
    
    h5 = (v + 4*u)/16 * EE.FourKronecker(Z,I,I,I)
    h6 = (v + 4*u)/16 * EE.FourKronecker(I,Z,I,I)
    h7 = (v + 4*u)/16 * EE.FourKronecker(I,I,Z,I)
    h8 = (v + 4*u)/16 * EE.FourKronecker(I,I,I,Z)
    
    h9 =  (v + 4*u)/16 * EE.FourKronecker(Z,Z,I,I)
    h10 = (v + 4*u)/16 * EE.FourKronecker(I,I,Z,Z)
    h11 = (v + 4*u)/16 * EE.FourKronecker(Z,I,Z,I)
    h12 = (v + 4*u)/16 * EE.FourKronecker(I,Z,I,Z)
    
    h13 = v/16 * EE.FourKronecker(Z,I,I,Z)
    h14 = v/16 * EE.FourKronecker(I,Z,Z,I)
    
    h15 = v/16 * EE.FourKronecker(Z,Z,Z,I)
    h16 = v/16 * EE.FourKronecker(Z,Z,I,Z)
    h17 = v/16 * EE.FourKronecker(Z,I,Z,Z)
    h18 = v/16 * EE.FourKronecker(I,Z,Z,Z)
    
    h19 = (v + 4*u)/16 * EE.FourKronecker(Z,Z,Z,Z)
         
    H = h0 + h1 + h2 + h3 + h4 + h5 + h6 + h7 + h8 + h9 + h10 + h11 + h12 + h13 + h14 + h15 + h16 + h17 + h18 + h19
    
    return expm(- 1j * H * delta_t)

def ExactTimeEvolutionTrotterOne(u,v,t,delta_t):
    '''
    Function to use numpy to calculate the 'exact' trotterized matrix exponential to calculate the e^(- i H t) unitary time evolution operator.

    Parameters
    ----------
    u : float
        One the parameters in the Hamiltonian, see report for description.
    V : float
        One the parameters in the Hamiltonian, see report for description.
    t : float
        One the parameters in the Hamiltonian, see report for description.
    delta_t : float
        The amount of time the operator evolves the system by.

    Returns
    -------
    numpy array (2 dim)
        numpy array describing the matrix exponential.

    '''
    X = np.array([[0,1],[1, 0]])
    Z = np.array([[1,0],[0,-1]])
    I = np.array([[1,0],[0, 1]])
    
    h0 = (8*t + 3 * u / 4 + v / 16) * EE.FourKronecker(I, I, I, I)
    
    h1 = -2*t * EE.FourKronecker(X,I,I,I)
    h2 = -2*t * EE.FourKronecker(I,X,I,I)
    h3 = -2*t * EE.FourKronecker(I,I,X,I)
    h4 = -2*t * EE.FourKronecker(I,I,I,X)
    
    h5 = (v + 4*u)/16 * EE.FourKronecker(Z,I,I,I)
    h6 = (v + 4*u)/16 * EE.FourKronecker(I,Z,I,I)
    h7 = (v + 4*u)/16 * EE.FourKronecker(I,I,Z,I)
    h8 = (v + 4*u)/16 * EE.FourKronecker(I,I,I,Z)
    
    h9 =  (v + 4*u)/16 * EE.FourKronecker(Z,Z,I,I)
    h10 = (v + 4*u)/16 * EE.FourKronecker(I,I,Z,Z)
    h11 = (v + 4*u)/16 * EE.FourKronecker(Z,I,Z,I)
    h12 = (v + 4*u)/16 * EE.FourKronecker(I,Z,I,Z)
    
    h13 = v/16 * EE.FourKronecker(Z,I,I,Z)
    h14 = v/16 * EE.FourKronecker(I,Z,Z,I)
    
    h15 = v/16 * EE.FourKronecker(Z,Z,Z,I)
    h16 = v/16 * EE.FourKronecker(Z,Z,I,Z)
    h17 = v/16 * EE.FourKronecker(Z,I,Z,Z)
    h18 = v/16 * EE.FourKronecker(I,Z,Z,Z)
    
    h19 = (v + 4*u)/16 * EE.FourKronecker(Z,Z,Z,Z)
            
    array = expm(- 1j * h0 * delta_t)
    array = array.dot(expm(- 1j * h1 * delta_t))
    array = array.dot(expm(- 1j * h2 * delta_t))
    array = array.dot(expm(- 1j * h3 * delta_t))
    array = array.dot(expm(- 1j * h4 * delta_t))
    array = array.dot(expm(- 1j * h5 * delta_t))
    array = array.dot(expm(- 1j * h6 * delta_t))
    array = array.dot(expm(- 1j * h7 * delta_t))
    array = array.dot(expm(- 1j * h8 * delta_t))
    array = array.dot(expm(- 1j * h9 * delta_t))
    array = array.dot(expm(- 1j * h10 * delta_t))
    array = array.dot(expm(- 1j * h11 * delta_t))
    array = array.dot(expm(- 1j * h12 * delta_t))
    array = array.dot(expm(- 1j * h13 * delta_t))
    array = array.dot(expm(- 1j * h14 * delta_t))
    array = array.dot(expm(- 1j * h15 * delta_t))
    array = array.dot(expm(- 1j * h16 * delta_t))
    array = array.dot(expm(- 1j * h17 * delta_t))
    array = array.dot(expm(- 1j * h18 * delta_t))
    array = array.dot(expm(- 1j * h19 * delta_t))

    return array
    
        
class ExactEvolution(c.Gate):
    '''
    Gate that implemts time evolution by the exact matrix exponential. Initializes with hamiltonian parameters u,v,t and angles phi,theta. Inherits c.Gate
    '''
    def _num_qubits_(self):
        return 4
    
    def _unitary_(self):
        return ExactTimeEvolution(self.u,self.v,self.t,self.delta_t)
        
    def __init__(self, u, v, t, delta_t):
        self.u = u
        self.v = v
        self.t = t
        self.delta_t = delta_t
        
class ExactEvolutionTrotterOne(c.Gate):
    '''
    Gate that implemts time evolution by the trotterized matrix exponential. Initializes with hamiltonian parameters u,v,t and angles phi,theta. Inherits c.Gate
    '''
    def _num_qubits_(self):
        return 4
    
    def _unitary_(self):
        return ExactTimeEvolutionTrotterOne(self.u,self.v,self.t,self.delta_t)
        
    def __init__(self, u, v, t, delta_t):
        self.u = u
        self.v = v
        self.t = t
        self.delta_t = delta_t
        
        
    