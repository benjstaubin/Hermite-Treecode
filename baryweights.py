# -*- coding: utf-8 -*-
"""
This function calculates barycentric weights given the nodes and integer
confluence s. This is adapted from "genbaryweights.m," a matlab file created
by Robert M. Corless.    

Ben St. Aubin 10/19/2018
"""
def baryweights(nodes, s_in):
    import numpy as np

    n = len(nodes)
    s = s_in * np.ones(n)
    
    n_diff = np.transpose(np.tile(nodes,(n,1)))-np.tile(nodes,(n,1))+np.eye(n)
    
    u = np.zeros((n, s_in))
    nu = np.zeros((n, s_in))
    nu[:,0]=1
    node_diff_recip = n_diff**-1

    for m in range(1,s_in):
        u[:,m]=sum(np.dot(np.diag(s),node_diff_recip**(m)-np.eye(n)))
        for i in range(n):
            nu[i,m]=(1/(m))*np.dot(u[i,0:m+1],np.transpose(np.flip(nu[i,0:m+1],0)))
            
    node_diff_recip = node_diff_recip**s_in
    
    w = np.flip(np.dot(np.diag(np.prod(node_diff_recip,axis = 1)),nu),1)
    
    return w
