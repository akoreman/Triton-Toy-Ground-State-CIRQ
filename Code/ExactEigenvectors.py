# -*- coding: utf-8 -*-
"""
Applied Quantum Algorithms Mini-Project 1, Tim Koreman (2418541), Leiden University (2020).

Function to calculate the exact eigenvectors and values of our hamiltonian.
"""
import numpy as np

def ExactEigenVectors(u,v,t):
    '''
    Function to calculate the eigenvalues and eigenvectors of the Hamiltonian using numpy.

    Parameters
    ----------
    v : float
        One the parameters in the Hamiltonian, see report for description.
    v : float
        One the parameters in the Hamiltonian, see report for description.
    t : float
        One the parameters in the Hamiltonian, see report for description.

    Returns
    -------
    Numpy array (2 dim)
        numpy array where the first element is the ordered list of eigenvalues and the second element contains the corresponding normalised eigenvectors.
        
    '''
    # definition of the 3 pauli matrices as np.array.
    X = np.array([[0,1],[1, 0]])
    Z = np.array([[1,0],[0,-1]])
    I = np.array([[1,0],[0, 1]])
    
    h0 = (8*t + 3 * u / 4 + v / 16) * FourKronecker(I, I, I, I)
    
    h1 = -2*t * FourKronecker(X,I,I,I)
    h2 = -2*t * FourKronecker(I,X,I,I)
    h3 = -2*t * FourKronecker(I,I,X,I)
    h4 = -2*t * FourKronecker(I,I,I,X)
    
    h5 = (v + 4*u)/16 * FourKronecker(Z,I,I,I)
    h6 = (v + 4*u)/16 * FourKronecker(I,Z,I,I)
    h7 = (v + 4*u)/16 * FourKronecker(I,I,Z,I)
    h8 = (v + 4*u)/16 * FourKronecker(I,I,I,Z)
    
    h9  = (v + 4*u)/16 * FourKronecker(Z,Z,I,I)
    h10 = (v + 4*u)/16 * FourKronecker(I,I,Z,Z)
    h11 = (v + 4*u)/16 * FourKronecker(Z,I,Z,I)
    h12 = (v + 4*u)/16 * FourKronecker(I,Z,I,Z)
    
    h13 = v/16 * FourKronecker(Z,I,I,Z)
    h14 = v/16 * FourKronecker(I,Z,Z,I)
    
    h15 = v/16 * FourKronecker(Z,Z,Z,I)
    h16 = v/16 * FourKronecker(Z,Z,I,Z)
    h17 = v/16 * FourKronecker(Z,I,Z,Z)
    h18 = v/16 * FourKronecker(I,Z,Z,Z)
    
    h19 = (v + 4*u)/16 * FourKronecker(Z,Z,Z,Z)
          
    H = h0 + h1 + h2 + h3 + h4 + h5 + h6 + h7 + h8 + h9 + h10 + h11 + h12 + h13 + h14 + h15 + h16 + h17 + h18 + h19

    return np.linalg.eigh(H)

def FourKronecker(t1, t2, t3, t4):
    '''
    Function to return the Kronecker product of four tenrsors. Note that this product is in general non-commutative.

    Parameters
    ----------
    t1 : np.array
        The first tensor.
    t2 : np.array
        The second tensor.
    t3 : np.array
        The third tensor.
    t4 : np.array
        The fourth tensor.

    Returns
    -------
    np.array
        The tensor describing the Kronecker product of the four tensors..

    '''
    t12 = np.kron(t1, t2)
    t34 = np.kron(t3,t4)
    
    return np.kron(t12, t34)