# -*- coding: utf-8 -*-
"""
These are functions created to calculate the l_i and l_i_prime terms for 
lagrange and hermite interpolation. 

Last update: 11/11/2018
Ben St. Aubin
"""
import numpy as np

def l_i(x,t_data,i):
    #If x is an array, returns an array of values, where out[n] = l_i(x[n])
    #
    #If x is a single value, returns a single value l_i(x)
    t = np.delete(t_data,i) 
    if np.size(x)==1:
        return np.product(x-t)/np.product(t_data[i]-t)
    else:
        num = np.subtract.outer(x,t)
        return np.product(num,axis=1)/np.product(t_data[i]-t)
    
def l_i_array(x,t_data):
    #if x is a single value, outputs a 1-D array "out", where out[i] = l_i(x) 
    #
    #if x is an array, outputs a 2-D array "out", where out[i,n] is l_i(x[n])
    if np.size(x)==1:    
        out = np.zeros(len(t_data))
        for i in range(len(t_data)):
            t = np.delete(t_data,i) 
            out[i] = np.product(x-t)/np.product(t_data[i]-t)
    else:
        out = np.zeros((len(t_data),len(x)))
        for i in range(len(t_data)):
            t = np.delete(t_data,i) 
            num = np.subtract.outer(x,t)
            out[i,:] = np.product(num,axis=1)/np.product(t_data[i]-t)
    return out

def l_i_p_array(t_nodes):
    #output is "l_prime", where l_prime[i] = l_i_prime(t_i), for t_i nodes
    x = np.array(t_nodes,float)
    l_prime = np.zeros(len(x))
    for i in range(len(x)):
        t = np.delete(t_nodes,i)
        l_prime[i] = np.sum(1/(t_nodes[i]-t))
    return l_prime

def l_i_prime(t_nodes,i):
    #output is l_i_prime(t_i) for t_i nodes input
    x = np.delete(t_nodes,i)
    return np.sum(1/(t_nodes[i]-x))