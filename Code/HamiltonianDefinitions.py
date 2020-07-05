# -*- coding: utf-8 -*-
"""
Applied Quantum Algorithms Mini-Project 1, Tim Koreman (2418541), Leiden University (2020).

Definiton of the Hamtiltonian as defined by Roggero et al. (2019). Encoding of the parity of the term with an ancilla and measuring that.
"""

import cirq as c

def H1(qubits, ancilla):
    yield c.H(qubits[0]) #measure in the eigenstates of X1
    yield c.CNOT(qubits[0], ancilla) 
    yield c.measure(ancilla, key='Zmeasurements')
               
def H2(qubits, ancilla):   
    yield c.H(qubits[1]) #measure in the eigenstates of X2
    yield c.CNOT(qubits[1], ancilla)  
    yield c.measure(ancilla, key='Zmeasurements')    
              
def H3(qubits, ancilla):
    yield c.H(qubits[2]) #measure in the eigenstates of X3
    yield c.CNOT(qubits[2], ancilla)    
    yield c.measure(ancilla, key='Zmeasurements')      
                
def H4(qubits, ancilla):
    yield c.H(qubits[3]) #measure in the eigenstates of X4
    yield c.CNOT(qubits[3], ancilla)  
    yield c.measure(ancilla, key='Zmeasurements') 
            
def H5(qubits, ancilla):
    yield c.CNOT(qubits[0], ancilla) 
    yield c.CNOT(qubits[3], ancilla)  
    yield c.measure(ancilla, key='Zmeasurements') 
        
def H6(qubits, ancilla):
    yield c.CNOT(qubits[1], ancilla)  
    yield c.CNOT(qubits[2], ancilla)    
    yield c.measure(ancilla, key='Zmeasurements') 
        
def H7(qubits, ancilla):
    yield c.CNOT(qubits[0], ancilla) 
    yield c.CNOT(qubits[1], ancilla)  
    yield c.CNOT(qubits[2], ancilla)  
    yield c.measure(ancilla, key='Zmeasurements') 
        
def H8(qubits, ancilla):
    yield c.CNOT(qubits[0], ancilla) 
    yield c.CNOT(qubits[1], ancilla)   
    yield c.CNOT(qubits[3], ancilla)  
    yield c.measure(ancilla, key='Zmeasurements') 
    
def H9(qubits, ancilla):
    yield c.CNOT(qubits[0], ancilla)  
    yield c.CNOT(qubits[2], ancilla)  
    yield c.CNOT(qubits[3], ancilla)  
    yield c.measure(ancilla, key='Zmeasurements') 
    
def H10(qubits, ancilla):
    yield c.CNOT(qubits[1], ancilla)  
    yield c.CNOT(qubits[2], ancilla)  
    yield c.CNOT(qubits[3], ancilla)  
    yield c.measure(ancilla, key='Zmeasurements') 
        
def H11(qubits, ancilla):
    yield c.CNOT(qubits[0], ancilla) 
    yield c.CNOT(qubits[1], ancilla)  
    yield c.measure(ancilla, key='Zmeasurements') 
    
def H12(qubits, ancilla):
    yield c.CNOT(qubits[2], ancilla)  
    yield c.CNOT(qubits[3], ancilla)  
    yield c.measure(ancilla, key='Zmeasurements') 

def H13(qubits, ancilla):
    yield c.CNOT(qubits[1], ancilla)  
    yield c.CNOT(qubits[3], ancilla)  
    yield c.measure(ancilla, key='Zmeasurements') 
    
def H14(qubits, ancilla):
    yield c.CNOT(qubits[0], ancilla)  
    yield c.CNOT(qubits[2], ancilla)  
    yield c.measure(ancilla, key='Zmeasurements') 
    
def H15(qubits, ancilla):   
    yield c.CNOT(qubits[0], ancilla) 
    yield c.measure(ancilla, key='Zmeasurements')
               
def H16(qubits, ancilla):   
    yield c.CNOT(qubits[1], ancilla)  
    yield c.measure(ancilla, key='Zmeasurements')    
              
def H17(qubits, ancilla): 
    yield c.CNOT(qubits[2], ancilla)  
    yield c.measure(ancilla, key='Zmeasurements')      
                
def H18(qubits, ancilla):
    yield c.CNOT(qubits[3], ancilla)  
    yield c.measure(ancilla, key='Zmeasurements')
    
def H19(qubits, ancilla):
    yield c.CNOT(qubits[0], ancilla) 
    yield c.CNOT(qubits[1], ancilla)  
    yield c.CNOT(qubits[2], ancilla)  
    yield c.CNOT(qubits[3], ancilla)  
    yield c.measure(ancilla, key='Zmeasurements')
    
    
    