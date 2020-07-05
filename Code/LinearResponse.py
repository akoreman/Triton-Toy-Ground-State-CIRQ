# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:44:25 2020

@author: Tim
"""

import openfermioncirq
import cirq as c
import openfermion as OF

import sympy
import numpy as np
import matplotlib.pyplot as plt

import random

import TimeEvolution as TE

import HamiltonianDefinitions as HD
import StatePreparation as SP
#import HamiltonianDefinitions as HamDef
'''
def TwoZGate(qubit1, qubit2, angle):
    yield c.CNOT(qubit1, qubit2)
    yield c.rz(angle)(qubit2)
    yield c.CNOT(qubit1, qubit2)
    
def ThreeZGate(qubit1, qubit2, qubit3, angle):
    yield c.CNOT(qubit1, qubit2)
    yield c.CNOT(qubit2, qubit3)
    yield c.rz(angle)(qubit3)
    yield c.CNOT(qubit2, qubit3)
    yield c.CNOT(qubit1, qubit2)
    
def FourZGate(qubit1, qubit2, qubit3, qubit4, angle):
    yield c.CNOT(qubit1, qubit2)
    yield c.CNOT(qubit2, qubit3)
    yield c.CNOT(qubit3, qubit4)
    yield c.rz(angle)(qubit3)
    yield c.CNOT(qubit3, qubit4)
    yield c.CNOT(qubit2, qubit3)
    yield c.CNOT(qubit1, qubit2)

def QPE(qubits, ancillaqubits, U, V,t, DeltaT):
    #ancillaqubits = [c.LineQubit(i + 4) for i in range(0,nAncilla)]
    #circ = c.Circuit()
               
    for i in range(0, len(ancillaqubits) ):
        yield c.H(ancillaqubits[i])
        
    for i in range(0, len(ancillaqubits) ):  
        #for j in range(0, 2**i):
        yield TE.TrotterizedEvolutionGate(U, V, t, DeltaT * 2**i).controlled()(ancillaqubits[i], *qubits)
            
    yield c.QFT(*ancillaqubits, inverse = True)       
    yield c.measure(*ancillaqubits, key='work') 

class TrotterizedEvolutionGateGamma(c.Gate): 
    def _num_qubits_(self):
        return 4
    
    def _decompose_(self, qubits): 
        yield c.rz(self.Angle)(qubits[2])
                     
    def __init__(self, gamma):     
        self.Angle = 0.5 * gamma
        
def FejerKernel(x,n):
    return 1/n * ((1 - np.cos(n*x))/(1 - np.cos(x)))

def ProbDist(y, W, coefflist, eigvallist):
    
    array = [ coefflist[i] * FejerKernel(2*np.pi*(eigvallist[i] - y/(2**W)), 2**W) for i in range(0, len(coefflist)) ]
    
    return 1/(2**W) * np.sum(array)
'''
W = 6
gamma = 0.1
DeltaT = 1
n_measurements =  10**6

U = -7
V = 28
t = 1

qubits = [c.LineQubit(1), c.LineQubit(2), c.LineQubit(3), c.LineQubit(4)]
ancillaqubit = c.LineQubit(0) 
workqubits = [c.LineQubit(i) for i in range(5, 5 + W)]

circ = c.Circuit()

circ.append(c.X(ancillaqubit))

circ.append(c.rz(np.pi /2)(ancillaqubit))
circ.append(c.H(ancillaqubit))

circ.append(c.CNOT(qubits[2], ancillaqubit))
circ.append(c.rz(gamma)(ancillaqubit))
circ.append(c.CNOT(qubits[2], ancillaqubit))
circ.append(c.H(ancillaqubit))
circ.append( c.rz( np.pi * 3/2)(ancillaqubit) )

circ.append(c.measure(ancillaqubit, key='ancilla') )


circ.append(SP.QPEStatePreparation(qubits, workqubits, U, V,t, DeltaT))

simulator = c.Simulator()
result = simulator.run(circ, repetitions=n_measurements)


ancillavalues = result.data["ancilla"].values 
indices = np.array(np.where(ancillavalues == 0))[0]

workvalues = result.data["work"].values 

workvalues = workvalues[indices]
length = len(workvalues)

probs = np.histogram(workvalues,50)[0]/length
omegasweep = np.linspace(0, 1, len(probs))

plt.plot(omegasweep, probs)
plt.yscale("log")

plt.xlabel(r'$\omega$')
plt.ylabel(r'$S_O (\omega)$')
plt.savefig('SO.pdf')

plt.show()

