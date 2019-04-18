# -*- coding: utf-8 -*-
"""
This calculates error between Hermite polynomial interpolation and Lagrange
interpolation in a given range for a given function. The error is calculated by
generating an interpolating polynomial and comparing values with the exact 
function along a uniform grid (spacing given by h). 

Ben St. Aubin
9/18/2018

Updated to add barycentric Hermite 10/23/2018 (also tweaked for readability)
"""
import numpy as np, matplotlib.pyplot as plt
from l_i_function import l_i, l_i_prime
from baryweights import baryweights

#The functions below describe our target function and its derivative
def f1(x):
    return x**20  

def f_prime1(x):
    return 20*x**19

def f2(x):
    return np.exp(x)  

def f_prime2(x):
    return np.exp(x)

def f3(x):
    return np.exp(-x**2) 

def f_prime3(x):
    return -2*x*np.exp(-x**2)

def f4(x):
    return 1/(1+16*x**2) 

def f_prime4(x):
    return -(32*x)/(1+16*x**2)**2

def f5(x):
    return np.exp(-x**(-2))

def f_prime5(x):
    return 2*x**(-3)*np.exp(-x**(-2))

def f6(x):
    return x**-1  

def f_prime6(x):
    return -x**-2

#number of interpolation points and interval length [a,b]
a = 2
b = 3
#h is spacing between points in the interval to plot function and interpolation
h = 0.01
iteration_num = 30

s = 7 #markersize in plots
r = 7 #bary markersize

#this initializes the lists to keep track of 2-norm error
hermite_error = []
lagrange_error = []
bary_error_h = []
bary_error_l = []
#calculating exact values
z = np.arange(a,b+h,h)
f_exact = f1(z)
    
for n in range(1,iteration_num+1):
    #creating chebyshev points
    c_nodes = 0.5*(a+b) + 0.5*(b-a)*np.cos((2*np.arange(1,n+1)-1)/(2*n)*np.pi)
    #c_nodes = np.linspace(a,b,n)
    
    #finding the "data" at the nodes
    f_c = f1(c_nodes)
    f_prime_c = f_prime1(c_nodes)
    
    #Hermite Interpolation
    herm_f = np.zeros(len(z))
    for i in range(n):
        herm_f+=(1-2*(z-c_nodes[i])*l_i_prime(c_nodes,i))*f_c[i]*(l_i(z,c_nodes,i))**2
        herm_f+=f_prime_c[i]*(z-c_nodes[i])*(l_i(z,c_nodes,i))**2
    
    #Lagrange Interpolation
    lag_f = np.zeros(len(z))
    for i in range(n):
        lag_f+=f_c[i]*l_i(z,c_nodes,i)
    
    #Barycentric Lagrange
    bary_p = np.zeros(len(z))
    bary_q = np.zeros(len(z))
    w = baryweights(c_nodes, 1)  
    for i in range(n):
        bary_p += (w[i,0]/(z-c_nodes[i]))*f_c[i]
        bary_q += w[i,0]/(z-c_nodes[i])
    bary_l = bary_p/bary_q

    #Barycentric Hermite        
    bary_p = np.zeros(len(z))
    bary_q = np.zeros(len(z))
    w = baryweights(c_nodes, 2)  
    for i in range(n):
        bary_p += ((w[i,0]/(z-c_nodes[i]))*f_c[i] +
                   (w[i,1]/(z-c_nodes[i])**2)*
                   (f_c[i] + f_prime_c[i]*(z-c_nodes[i])))
        bary_q += w[i,0]/(z-c_nodes[i]) + w[i,1]/(z-c_nodes[i])**2
    bary_h = bary_p/bary_q
        

    #error calculation (2-norm of relative error)
    error_herm = np.sqrt(np.sum((herm_f - f_exact)**2)/np.sum(f_exact**2))
    error_lag = np.sqrt(np.sum((lag_f - f_exact)**2)/np.sum(f_exact**2))   
    error_bary_h = np.sqrt(np.sum((bary_h - f_exact)**2)/np.sum(f_exact**2))
    error_bary_l = np.sqrt(np.sum((bary_l - f_exact)**2)/np.sum(f_exact**2))
    hermite_error.append(error_herm)
    lagrange_error.append(error_lag)
    bary_error_l.append(error_bary_l)
    bary_error_h.append(error_bary_h)
    
hermite_error1 = hermite_error
lagrange_error1 = lagrange_error
bary_error_l1 = bary_error_l
bary_error_h1 = bary_error_h



#this initializes the lists to keep track of 2-norm error
hermite_error = []
lagrange_error = []
bary_error_h = []
bary_error_l = []
#calculating exact values
f_exact = f2(z)
    
for n in range(1,iteration_num+1):
    #creating chebyshev points
    c_nodes = 0.5*(a+b) + 0.5*(b-a)*np.cos((2*np.arange(1,n+1)-1)/(2*n)*np.pi)
    #c_nodes = np.linspace(a,b,n)
    
    #finding the "data" at the nodes
    f_c = f2(c_nodes)
    f_prime_c = f_prime2(c_nodes)
    
    #Hermite Interpolation
    herm_f = np.zeros(len(z))
    for i in range(n):
        herm_f+=(1-2*(z-c_nodes[i])*l_i_prime(c_nodes,i))*f_c[i]*(l_i(z,c_nodes,i))**2
        herm_f+=f_prime_c[i]*(z-c_nodes[i])*(l_i(z,c_nodes,i))**2
    
    #Lagrange Interpolation
    lag_f = np.zeros(len(z))
    for i in range(n):
        lag_f+=f_c[i]*l_i(z,c_nodes,i)
    
    #Barycentric Lagrange
    bary_p = np.zeros(len(z))
    bary_q = np.zeros(len(z))
    w = baryweights(c_nodes, 1)  
    for i in range(n):
        bary_p += (w[i,0]/(z-c_nodes[i]))*f_c[i]
        bary_q += w[i,0]/(z-c_nodes[i])
    bary_l = bary_p/bary_q

    #Barycentric Hermite        
    bary_p = np.zeros(len(z))
    bary_q = np.zeros(len(z))
    w = baryweights(c_nodes, 2)  
    for i in range(n):
        bary_p += ((w[i,0]/(z-c_nodes[i]))*f_c[i] +
                   (w[i,1]/(z-c_nodes[i])**2)*
                   (f_c[i] + f_prime_c[i]*(z-c_nodes[i])))
        bary_q += w[i,0]/(z-c_nodes[i]) + w[i,1]/(z-c_nodes[i])**2
    bary_h = bary_p/bary_q
        

    #error calculation (2-norm of relative error)
    error_herm = np.sqrt(np.sum((herm_f - f_exact)**2)/np.sum(f_exact**2))
    error_lag = np.sqrt(np.sum((lag_f - f_exact)**2)/np.sum(f_exact**2))   
    error_bary_h = np.sqrt(np.sum((bary_h - f_exact)**2)/np.sum(f_exact**2))
    error_bary_l = np.sqrt(np.sum((bary_l - f_exact)**2)/np.sum(f_exact**2))
    hermite_error.append(error_herm)
    lagrange_error.append(error_lag)
    bary_error_l.append(error_bary_l)
    bary_error_h.append(error_bary_h)
    
hermite_error2 = hermite_error
lagrange_error2 = lagrange_error
bary_error_l2 = bary_error_l
bary_error_h2 = bary_error_h

#this initializes the lists to keep track of 2-norm error
hermite_error = []
lagrange_error = []
bary_error_h = []
bary_error_l = []
#calculating exact values
f_exact = f3(z)
    
for n in range(1,iteration_num+1):
    #creating chebyshev points
    c_nodes = 0.5*(a+b) + 0.5*(b-a)*np.cos((2*np.arange(1,n+1)-1)/(2*n)*np.pi)
    #c_nodes = np.linspace(a,b,n)
     
    #finding the "data" at the nodes
    f_c = f3(c_nodes)
    f_prime_c = f_prime3(c_nodes)
    
    #Hermite Interpolation
    herm_f = np.zeros(len(z))
    for i in range(n):
        herm_f+=(1-2*(z-c_nodes[i])*l_i_prime(c_nodes,i))*f_c[i]*(l_i(z,c_nodes,i))**2
        herm_f+=f_prime_c[i]*(z-c_nodes[i])*(l_i(z,c_nodes,i))**2
    
    #Lagrange Interpolation
    lag_f = np.zeros(len(z))
    for i in range(n):
        lag_f+=f_c[i]*l_i(z,c_nodes,i)
    
    #Barycentric Lagrange
    bary_p = np.zeros(len(z))
    bary_q = np.zeros(len(z))
    w = baryweights(c_nodes, 1)  
    for i in range(n):
        bary_p += (w[i,0]/(z-c_nodes[i]))*f_c[i]
        bary_q += w[i,0]/(z-c_nodes[i])
    bary_l = bary_p/bary_q

    #Barycentric Hermite        
    bary_p = np.zeros(len(z))
    bary_q = np.zeros(len(z))
    w = baryweights(c_nodes, 2)  
    for i in range(n):
        bary_p += ((w[i,0]/(z-c_nodes[i]))*f_c[i] +
                   (w[i,1]/(z-c_nodes[i])**2)*
                   (f_c[i] + f_prime_c[i]*(z-c_nodes[i])))
        bary_q += w[i,0]/(z-c_nodes[i]) + w[i,1]/(z-c_nodes[i])**2
    bary_h = bary_p/bary_q
        

    #error calculation (2-norm of relative error)
    error_herm = np.sqrt(np.sum((herm_f - f_exact)**2)/np.sum(f_exact**2))
    error_lag = np.sqrt(np.sum((lag_f - f_exact)**2)/np.sum(f_exact**2))   
    error_bary_h = np.sqrt(np.sum((bary_h - f_exact)**2)/np.sum(f_exact**2))
    error_bary_l = np.sqrt(np.sum((bary_l - f_exact)**2)/np.sum(f_exact**2))
    hermite_error.append(error_herm)
    lagrange_error.append(error_lag)
    bary_error_l.append(error_bary_l)
    bary_error_h.append(error_bary_h)
    
hermite_error3 = hermite_error
lagrange_error3 = lagrange_error
bary_error_l3 = bary_error_l
bary_error_h3 = bary_error_h

#this initializes the lists to keep track of 2-norm error
hermite_error = []
lagrange_error = []
bary_error_h = []
bary_error_l = []
#calculating exact values
f_exact = f4(z)
    
for n in range(1,iteration_num+1):
    #creating chebyshev points
    c_nodes = 0.5*(a+b) + 0.5*(b-a)*np.cos((2*np.arange(1,n+1)-1)/(2*n)*np.pi)
    #c_nodes = np.linspace(a,b,n)
     
    #finding the "data" at the nodes
    f_c = f4(c_nodes)
    f_prime_c = f_prime4(c_nodes)
    
    #Hermite Interpolation
    herm_f = np.zeros(len(z))
    for i in range(n):
        herm_f+=(1-2*(z-c_nodes[i])*l_i_prime(c_nodes,i))*f_c[i]*(l_i(z,c_nodes,i))**2
        herm_f+=f_prime_c[i]*(z-c_nodes[i])*(l_i(z,c_nodes,i))**2
    
    #Lagrange Interpolation
    lag_f = np.zeros(len(z))
    for i in range(n):
        lag_f+=f_c[i]*l_i(z,c_nodes,i)
    
    #Barycentric Lagrange
    bary_p = np.zeros(len(z))
    bary_q = np.zeros(len(z))
    w = baryweights(c_nodes, 1)  
    for i in range(n):
        bary_p += (w[i,0]/(z-c_nodes[i]))*f_c[i]
        bary_q += w[i,0]/(z-c_nodes[i])
    bary_l = bary_p/bary_q

    #Barycentric Hermite        
    bary_p = np.zeros(len(z))
    bary_q = np.zeros(len(z))
    w = baryweights(c_nodes, 2)  
    for i in range(n):
        bary_p += ((w[i,0]/(z-c_nodes[i]))*f_c[i] +
                   (w[i,1]/(z-c_nodes[i])**2)*
                   (f_c[i] + f_prime_c[i]*(z-c_nodes[i])))
        bary_q += w[i,0]/(z-c_nodes[i]) + w[i,1]/(z-c_nodes[i])**2
    bary_h = bary_p/bary_q
        

    #error calculation (2-norm of relative error)
    error_herm = np.sqrt(np.sum((herm_f - f_exact)**2)/np.sum(f_exact**2))
    error_lag = np.sqrt(np.sum((lag_f - f_exact)**2)/np.sum(f_exact**2))   
    error_bary_h = np.sqrt(np.sum((bary_h - f_exact)**2)/np.sum(f_exact**2))
    error_bary_l = np.sqrt(np.sum((bary_l - f_exact)**2)/np.sum(f_exact**2))
    hermite_error.append(error_herm)
    lagrange_error.append(error_lag)
    bary_error_l.append(error_bary_l)
    bary_error_h.append(error_bary_h)
    
hermite_error4 = hermite_error
lagrange_error4 = lagrange_error
bary_error_l4 = bary_error_l
bary_error_h4 = bary_error_h

#this initializes the lists to keep track of 2-norm error
hermite_error = []
lagrange_error = []
bary_error_h = []
bary_error_l = []
#calculating exact values
f_exact = f5(z)
    
for n in range(1,iteration_num+1):
    #creating chebyshev points
    c_nodes = 0.5*(a+b) + 0.5*(b-a)*np.cos((2*np.arange(1,n+1)-1)/(2*n)*np.pi)
    #c_nodes = np.linspace(a,b,n)
     
    #finding the "data" at the nodes
    f_c = f5(c_nodes)
    f_prime_c = f_prime5(c_nodes)
    
    #Hermite Interpolation
    herm_f = np.zeros(len(z))
    for i in range(n):
        herm_f+=(1-2*(z-c_nodes[i])*l_i_prime(c_nodes,i))*f_c[i]*(l_i(z,c_nodes,i))**2
        herm_f+=f_prime_c[i]*(z-c_nodes[i])*(l_i(z,c_nodes,i))**2
    
    #Lagrange Interpolation
    lag_f = np.zeros(len(z))
    for i in range(n):
        lag_f+=f_c[i]*l_i(z,c_nodes,i)
    
    #Barycentric Lagrange
    bary_p = np.zeros(len(z))
    bary_q = np.zeros(len(z))
    w = baryweights(c_nodes, 1)  
    for i in range(n):
        bary_p += (w[i,0]/(z-c_nodes[i]))*f_c[i]
        bary_q += w[i,0]/(z-c_nodes[i])
    bary_l = bary_p/bary_q

    #Barycentric Hermite        
    bary_p = np.zeros(len(z))
    bary_q = np.zeros(len(z))
    w = baryweights(c_nodes, 2)  
    for i in range(n):
        bary_p += ((w[i,0]/(z-c_nodes[i]))*f_c[i] +
                   (w[i,1]/(z-c_nodes[i])**2)*
                   (f_c[i] + f_prime_c[i]*(z-c_nodes[i])))
        bary_q += w[i,0]/(z-c_nodes[i]) + w[i,1]/(z-c_nodes[i])**2
    bary_h = bary_p/bary_q
        

    #error calculation (2-norm of relative error)
    error_herm = np.sqrt(np.sum((herm_f - f_exact)**2)/np.sum(f_exact**2))
    error_lag = np.sqrt(np.sum((lag_f - f_exact)**2)/np.sum(f_exact**2))   
    error_bary_h = np.sqrt(np.sum((bary_h - f_exact)**2)/np.sum(f_exact**2))
    error_bary_l = np.sqrt(np.sum((bary_l - f_exact)**2)/np.sum(f_exact**2))
    hermite_error.append(error_herm)
    lagrange_error.append(error_lag)
    bary_error_l.append(error_bary_l)
    bary_error_h.append(error_bary_h)
    
hermite_error5 = hermite_error
lagrange_error5 = lagrange_error
bary_error_l5 = bary_error_l
bary_error_h5 = bary_error_h

#this initializes the lists to keep track of 2-norm error
hermite_error = []
lagrange_error = []
bary_error_h = []
bary_error_l = []
#calculating exact values
f_exact = f6(z)
    
for n in range(1,iteration_num+1):
    #creating chebyshev points
    c_nodes = 0.5*(a+b) + 0.5*(b-a)*np.cos((2*np.arange(1,n+1)-1)/(2*n)*np.pi)
    #c_nodes = np.linspace(a,b,n)
     
    #finding the "data" at the nodes
    f_c = f6(c_nodes)
    f_prime_c = f_prime6(c_nodes)
    
    #Hermite Interpolation
    herm_f = np.zeros(len(z))
    for i in range(n):
        herm_f+=(1-2*(z-c_nodes[i])*l_i_prime(c_nodes,i))*f_c[i]*(l_i(z,c_nodes,i))**2
        herm_f+=f_prime_c[i]*(z-c_nodes[i])*(l_i(z,c_nodes,i))**2
    
    #Lagrange Interpolation
    lag_f = np.zeros(len(z))
    for i in range(n):
        lag_f+=f_c[i]*l_i(z,c_nodes,i)
    
    #Barycentric Lagrange
    bary_p = np.zeros(len(z))
    bary_q = np.zeros(len(z))
    w = baryweights(c_nodes, 1)  
    for i in range(n):
        bary_p += (w[i,0]/(z-c_nodes[i]))*f_c[i]
        bary_q += w[i,0]/(z-c_nodes[i])
    bary_l = bary_p/bary_q

    #Barycentric Hermite        
    bary_p = np.zeros(len(z))
    bary_q = np.zeros(len(z))
    w = baryweights(c_nodes, 2)  
    for i in range(n):
        bary_p += ((w[i,0]/(z-c_nodes[i]))*f_c[i] +
                   (w[i,1]/(z-c_nodes[i])**2)*
                   (f_c[i] + f_prime_c[i]*(z-c_nodes[i])))
        bary_q += w[i,0]/(z-c_nodes[i]) + w[i,1]/(z-c_nodes[i])**2
    bary_h = bary_p/bary_q
        

    #error calculation (2-norm of relative error)
    error_herm = np.sqrt(np.sum((herm_f - f_exact)**2)/np.sum(f_exact**2))
    error_lag = np.sqrt(np.sum((lag_f - f_exact)**2)/np.sum(f_exact**2))   
    error_bary_h = np.sqrt(np.sum((bary_h - f_exact)**2)/np.sum(f_exact**2))
    error_bary_l = np.sqrt(np.sum((bary_l - f_exact)**2)/np.sum(f_exact**2))
    hermite_error.append(error_herm)
    lagrange_error.append(error_lag)
    bary_error_l.append(error_bary_l)
    bary_error_h.append(error_bary_h)
    
hermite_error6 = hermite_error
lagrange_error6 = lagrange_error
bary_error_l6 = bary_error_l
bary_error_h6 = bary_error_h
    
    
#Making the plots 
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2)
x = np.arange(1,iteration_num+1)


ax1.semilogy(x,lagrange_error1,'b^', label = 'Standard Lagrange', markersize = s)
ax1.semilogy(x,bary_error_l1,'m+', label = 'Barycentric Lagrange', markersize = r)
ax1.semilogy(x,hermite_error1, 'ro', label = 'Standard Hermite', markersize = s)
ax1.semilogy(x,bary_error_h1,'gx', label = 'Barycentric Hermite', markersize = r)
#ax1.set_xlabel("n")
#ax1.set_ylabel("Relative 2-Norm Error")
#ax1.legend()
ax1.text(0.7, 0.75, r'$x^{20}$', fontsize = 15, transform = ax1.transAxes)
ax1.grid(b=None, which='major', axis='both',linestyle='--')

ax2.semilogy(x,lagrange_error2,'b^', label = 'Standard Lagrange', markersize = s)
ax2.semilogy(x,bary_error_l2,'m+', label = 'Barycentric Lagrange', markersize = r)
ax2.semilogy(x,hermite_error2, 'ro', label = 'Standard Hermite', markersize = s)
ax2.semilogy(x,bary_error_h2,'gx', label = 'Barycentric Hermite', markersize = r)
#ax2.set_xlabel("n")
#ax2.set_ylabel("Relative 2-Norm Error")
#ax2.legend()
ax2.text(0.7, 0.75, r'$e^x$', fontsize = 15, transform = ax2.transAxes)
ax2.grid(b=None, which='major', axis='both',linestyle='--')

ax3.semilogy(x,lagrange_error3,'b^', label = 'Standard Lagrange', markersize = s)
ax3.semilogy(x,bary_error_l3,'m+', label = 'Barycentric Lagrange', markersize = r)
ax3.semilogy(x,hermite_error3, 'ro', label = 'Standard Hermite', markersize = s)
ax3.semilogy(x,bary_error_h3,'gx', label = 'Barycentric Hermite', markersize = r)
#ax3.set_xlabel("n")
ax3.set_ylabel("Relative 2-Norm Error")
#ax3.legend()
ax3.text(0.7, 0.75, r'$e^{-x^2}$', fontsize = 15, transform = ax3.transAxes)
ax3.grid(b=None, which='major', axis='both',linestyle='--')

ax4.semilogy(x,lagrange_error4,'b^', label = 'Standard Lagrange', markersize = s)
ax4.semilogy(x,bary_error_l4,'m+', label = 'Barycentric Lagrange', markersize = r)
ax4.semilogy(x,hermite_error4, 'ro', label = 'Standard Hermite', markersize = s)
ax4.semilogy(x,bary_error_h4,'gx', label = 'Barycentric Hermite', markersize = r)
#ax4.set_xlabel("n")
#ax4.set_ylabel("Relative 2-Norm Error")
#ax4.legend()
ax4.text(0.7, 0.75, r'$\frac{1}{1+16x^2}$', fontsize = 15, transform = ax4.transAxes)
ax4.grid(b=None, which='major', axis='both',linestyle='--')

ax5.semilogy(x,lagrange_error5,'b^', label = 'Standard Lagrange', markersize = s)
ax5.semilogy(x,bary_error_l5,'m+', label = 'Barycentric Lagrange', markersize = r)
ax5.semilogy(x,hermite_error5, 'ro', label = 'Standard Hermite', markersize = s)
ax5.semilogy(x,bary_error_h5,'gx', label = 'Barycentric Hermite', markersize = r)
ax5.set_xlabel("                                                                                            Order of interpolation, p")
#ax5.set_ylabel("Relative 2-Norm Error")
#ax5.legend()
ax5.text(0.7, 0.75, r'$e^\frac{-1}{x^2}$', fontsize = 15, transform = ax5.transAxes)
ax5.grid(b=None, which='major', axis='both',linestyle='--')

ax6.semilogy(x,lagrange_error6,'b^', label = 'Standard Lagrange', markersize = s)
ax6.semilogy(x,bary_error_l6,'m+', label = 'Barycentric Lagrange', markersize = r)
ax6.semilogy(x,hermite_error6, 'ro', label = 'Standard Hermite', markersize = s)
ax6.semilogy(x,bary_error_h6,'gx', label = 'Barycentric Hermite', markersize = r)
#ax6.set_xlabel("p")
#ax6.set_ylabel("Relative 2-Norm Error")
#ax6.legend()
ax6.text(0.7, 0.75, r'$\frac{1}{x}$', fontsize = 15, transform = ax6.transAxes)
ax6.grid(b=None, which='major', axis='both',linestyle='--')


plt.savefig('subplots.png')

