# -*- coding: utf-8 -*-
"""
This file automatically generates random data and weights and performs a 
treecode approximation for the 1/r kernel function. During the "compute
interaction" stage, multiprocessing is performed by spawning c-2 processes, 
where c is the automatically detected amount of physical CPU cores in your 
system. The tree structure, weights, and data are stored to the directory this 
file is placed in and deleted when the program is complete.

Adjustable parameters:
MAC parameter (Controls target and source separation condition... If MAC
               is satisfied, a far-field expansion will be performed to 
               approximate interaction between target and source cluster.)
Order of approximation p
Size of generated data N
max leaf size n
Far-Field approximation method (Hermite or Lagrange)

When collecting data sets, I add loops over various parameters. I point out
places in the code where I add these loops. 

Ben St. Aubin
4/6/2019
"""
import numpy as np
import multiprocessing as mp
import pickle, os, time

### This class is used to store data and attributes for each panel
class Panel_3_D:
    '''Input data as array, start and end as array or list, level as int'''
    def __init__(self, indices, start, end, level = 0, leaf = False):
        self.indices = np.array(indices, int)
        self.low = np.array(start)
        self.high = np.array(end)
        self.level = int(level)
        self.center = (self.low + self.high)/2
        self.radius = np.linalg.norm(self.high - self.center) 
        self.numpoints = len(indices)
        self.leaf = leaf
        #self.nodes = None

### The functions below describe our target function and its derivatives
def f(array):
    r = np.sqrt(np.einsum('ij,ij->j',array,array))  
    return r**(-1)

def ft(array,method):
    r = np.sqrt(np.einsum('ij,ij->j',array,array))
    if method == 'lagrange': 
        return r**(-1)
    else: #In this case, the method must be hermite    
        x = array[0,:] 
        y = array[1,:] 
        z = array[2,:]
        f = r**(-1)
        f_px = x*r**(-3)
        f_py = y*r**(-3)
        f_pz = z*r**(-3)
        f_pyz = 3*y*z*r**(-5)
        f_pxz = 3*x*z*r**(-5)
        f_pxy = 3*x*y*r**(-5)
        f_pxyz = 15*x*y*z*r**(-7)
        return np.array([f, f_pz, f_py, f_px, f_pyz, f_pxz, f_pxy, f_pxyz])     

def l_i_array(x,t_data):
    #if x is a single value, outputs a 1-D array "out", where out[i] = l_i(x) 
    #elif x is an array, outputs a 2-D array "out", where out[i,n] is l_i(x[n])
    if np.size(x)==1:    
        out = np.zeros(len(t_data))
        for i in range(len(t_data)):
            t = np.delete(t_data,i) 
            out[i] = np.product(x - t)/np.product(t_data[i] - t)
    else:
        out = np.zeros((len(t_data),len(x)))
        for i in range(len(t_data)):
            t = np.delete(t_data,i) 
            num = np.subtract.outer(x ,t)
            out[i,:] = np.product(num, axis=1)/np.product(t_data[i] - t)
    return out

def l_i_p_array(t_nodes):
    #output is "l_prime", where l_prime[i] = l_i_prime(t_i), for t_i nodes
    x = np.array(t_nodes,float)
    l_prime = np.zeros(len(x))
    for i in range(len(x)):
        t = np.delete(t_nodes, i)
        l_prime[i] = np.sum(1/(t_nodes[i] - t))
    return l_prime

### This function will split an array based on a condition. Used to build tree
def split(array, condition):
    #This splits an array based on a condition
    return [array[:, condition], array[:, ~condition]]

### This function splits a 3D panel into children
def panel_split_3_D(panel,maxpt):
    #Splits a 3-D panel into eight, based on the center point.   
    current = np.vstack((points[:,panel.indices],panel.indices))
    low = panel.low
    high = panel.high
    cen = panel.center
    lev = panel.level + 1    
    #finding relevant low and high points for split data
    b_low = np.array([cen[0],low[1],low[2]])
    b_high = np.array([high[0],cen[1],cen[2]]) 
    c_low = np.array([low[0],cen[1],low[2]])
    c_high = np.array([cen[0],high[1],cen[2]])
    d_low = np.array([cen[0],cen[1],low[2]])
    d_high = np.array([high[0],high[1],cen[2]])
    e_low = np.array([low[0],low[1],cen[2]])
    e_high = np.array([cen[0],cen[1],high[2]])
    f_low = np.array([cen[0],low[1],cen[2]])
    f_high = np.array([high[0],cen[1],high[2]])
    g_low = np.array([low[0],cen[1],cen[2]])
    g_high = np.array([cen[0],high[1],high[2]]) 
    #split in x direction
    new1 , new2 = split(current, current[0,:]<cen[0])
    #split in y direction
    new3 , new5 = split(new1, new1[1,:]<cen[1])
    new4 , new6 = split(new2, new2[1,:]<cen[1])   
    #Split in z direction
    A, E = split(new3, new3[2,:]<cen[2])
    B, F = split(new4, new4[2,:]<cen[2])
    C, G = split(new5, new5[2,:]<cen[2])
    D, H = split(new6, new6[2,:]<cen[2])
    #Returns children; the last Panel argument returns boolean for leaf status
    return(Panel_3_D(A[3], low,   cen,    lev, len(A[3])<maxpt),
           Panel_3_D(B[3], b_low, b_high, lev, len(B[3])<maxpt),
           Panel_3_D(C[3], c_low, c_high, lev, len(C[3])<maxpt),
           Panel_3_D(D[3], d_low, d_high, lev, len(D[3])<maxpt),
           Panel_3_D(E[3], e_low, e_high, lev, len(E[3])<maxpt),
           Panel_3_D(F[3], f_low, f_high, lev, len(F[3])<maxpt),
           Panel_3_D(G[3], g_low, g_high, lev, len(G[3])<maxpt),
           Panel_3_D(H[3], cen,   high,   lev, len(H[3])<maxpt))  

###This is an iterative function, used to approximate interactions
def comp_interaction(target_index, panel, result = 0):
    # This condition is my optimization step
    if (panel.numpoints < p**3 and method == 'lagrange'):
        return direct_sum(target_index,panel)
    if (panel.numpoints < 8*p**3 and method == 'hermite'):
        return direct_sum(target_index,panel) 
    
    # Below are the "standard" compute interaction steps   
    dist = np.linalg.norm(points[:,target_index] - panel.center)    
    if panel.radius/dist < MAC: #target and source are well-separated if so
        return far_field_expansion(target_index,panel)
    elif panel.leaf == True: #in this case, there are no further levels
        return direct_sum(target_index,panel)
    else: #if none of the checks passed, this routine is called for children
        child_level = panel.level + 1
        children_indices = panel.children
        for c_i in children_indices:
            result += comp_interaction(target_index,tree[child_level][c_i])
    return result            
   
### This function performs a far-field expansion
def far_field_expansion(target_index, panel):    
    if method == 'lagrange':
        L = panel.far_field_sum
        f_values = ft(np.vstack(points[:,target_index]) - panel.nodes, method)
        return np.sum(L*f_values)                       
    if method == 'hermite':
        Moments = panel.far_field_sum
        f_values = ft(np.vstack(points[:,target_index]) - panel.nodes, method)
        return np.sum(Moments*f_values)            

### This Function performs a direct summation for a target and source points
def direct_sum(target_index, panel):
    # If a panel is empty, return 0
    if panel.numpoints == 0: return 0    
    # The following removes targ point if it is contained in the leaf
    mask = panel.indices[panel.indices != target_index]
    return np.sum(weights[mask] * f(np.vstack(points[:,target_index]) - 
                  points[:,mask]))

def chunker(chunk):
    #This function acts as an intermediary between the multiprocessing pool and
    #the comp_interaction function. This function is necessary because each new
    #instance must load in the tree, data, and some constants.
    global tree, MAC, points, method, weights, p    
    points = np.load('data_3D.npy')
    weights = np.load('weights_3D.npy')
    with open('tree_file.pickle', 'rb') as handle: tree = pickle.load(handle)
    MAC = tree['MAC']   
    method = tree['method']  
    p = tree['node_num']
    result = np.zeros(len(chunk))
    for j in range(len(chunk)): result[j] = comp_interaction(chunk[j],tree[0][1])
    return result

def direct_chunker(chunk):
    #Given an iterable input of indices, this outputs the direct sum for those
    #indices over the entire set of points. This can be called by a pool for 
    #parallel computing after breaking up the points by index.
    global tree, points, weights
    with open('tree_file.pickle', 'rb') as handle: tree = pickle.load(handle)
    points = np.load('data_3D.npy')
    weights = np.load('weights_3D.npy')
    result = np.zeros(len(chunk))
    for j in range(len(chunk)): 
        result[j] = direct_sum(chunk[j], tree[0][1])
    return result
    
if __name__ == '__main__':
    N_op = [10000, 31623, 100000, 316238, 1000000, 3162378]
    MAC_op = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    node_op = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    # The 'if' statement is needed so this section is not run during
    # the multiprocessing portion of the code (compute interaction). If this 
    # statement were not here, this section of the code would be run by every
    # process spawned instead of just once in the main file.
    
    #--------------YOU CAN ADD A LOOP HERE OVER N_op---------------------
       
    ###TREECODE PARAMETERS
    #method = 'lagrange' 
    method = 'hermite'
    num_points = N_op[0]# The number of data points    
    p = node_op[6]      # adjuts order of interpolation
    MAC = MAC_op[7]     # lower gives higher accuracy but takes more time
    n = 1000            # n is the max amount of points per leaf       
           
    ###Random Data Creation 
    a = 0 #lower cube boundary (3D space)
    b = 1 #upper cube boundary (3D space)     
    weights = np.random.uniform(-1,1,num_points)

    #points uniform on sphere - comment out if you want to test points in cube
    radius = (a+b)/2 
    circle_points = np.random.randn(3, num_points)
    circle_points *=  (radius / np.linalg.norm(circle_points, axis=0))
    x = circle_points[0]+(a+b)/2
    y = circle_points[1]+(a+b)/2
    z = circle_points[2]+(a+b)/2
    
    """
    #points uniform in cube - comment out if you want to test points on sphere
    x = np.random.uniform(a,b,num_points)
    y = np.random.uniform(a,b,num_points)
    z = np.random.uniform(a,b,num_points)  
    """    
    
    points = np.array([x,y,z]) 
    indices = np.arange(num_points) #to keep track of point indices in tree
    np.save('data_3D.npy',points) #saves to file to access later
    np.save('weights_3D.npy',weights) #saves to file to access later

    
    ### The following initialization and loop constructs the tree
    low_corner = [a,a,a] #low corner boundry of cube containing points
    high_corner = [b,b,b] #upper corner boundry of cube containing points    
    init_pan = Panel_3_D(indices,low_corner,high_corner,0)  
    print('building tree, N=',num_points)
    tree = {0:{1 : init_pan}}
    tree_temp = {}
    level = 0
    while any(x.leaf == False for x in tree[level].values()) == True:
        index = 1
        for panel in tree[level].values():
            if panel.numpoints >= n:
                (tree_temp[index],   tree_temp[index+1], tree_temp[index+2], 
                 tree_temp[index+3], tree_temp[index+4], tree_temp[index+5],
                 tree_temp[index+6], tree_temp[index+7]) = panel_split_3_D(panel,n)
                panel.children = (index, index+1, index+2, index+3, index+4,
                                  index+5, index+6, index+7)
                index += 8
        tree[level+1] = dict(tree_temp)
        level += 1
        tree_temp = {}

    #--------------YOU CAN ADD A LOOP HERE OVER node_op--------------------
    tree_time_start = time.time()
    
    ### This calculates far-field components for each panel of the tree
    print('calculating far-field components, p =',p)       
    level_count = len(tree)-1
    current_level = 1
    while current_level <= level_count:
        for panel in tree[current_level].values():
            if (panel.numpoints < p**3 and method == 'lagrange'):
                continue
            if (panel.numpoints < 8*p**3 and method == 'hermite'):
                continue           
            #source interval
            ax = panel.low[0]
            bx = panel.high[0]
            ay = panel.low[1]
            by = panel.high[1]
            az = panel.low[2]
            bz = panel.high[2]
            mask = panel.indices
            panel_data = points[:,mask]
            panel_weights = weights[mask]
            y_x = panel_data[0,:]
            y_y = panel_data[1,:]
            y_z = panel_data[2,:]        
            #generate chebychev nodes and store to panel
            tx = 0.5*(ax+bx) + 0.5*(bx-ax)*np.cos((2*np.arange(1,p+1)-1)/(2*p)*np.pi)
            ty = 0.5*(ay+by) + 0.5*(by-ay)*np.cos((2*np.arange(1,p+1)-1)/(2*p)*np.pi)        
            tz = 0.5*(az+bz) + 0.5*(bz-az)*np.cos((2*np.arange(1,p+1)-1)/(2*p)*np.pi)        
            cx,cy,cz = np.meshgrid(tx,ty,tz, indexing='ij')
            panel.nodes = np.array([np.ravel(cx), np.ravel(cy), np.ravel(cz)])
            
            if method == 'lagrange':
                l_i_y_x = l_i_array(y_x,tx)
                l_i_y_y = l_i_array(y_y,ty)
                l_i_y_z = l_i_array(y_z,tz)    
                L = np.einsum('m,im,jm,km->ijk',
                              panel_weights,l_i_y_x,l_i_y_y,l_i_y_z) 
                panel.far_field_sum = np.ravel(L) 
                
            if method == 'hermite':            
                l_i_y_x = l_i_array(y_x,tx)
                l_i_y_y = l_i_array(y_y,ty)
                l_i_y_z = l_i_array(y_z,tz)
                l_px = l_i_p_array(tx)    
                l_py = l_i_p_array(ty)
                l_pz = l_i_p_array(tz)
                h_i_1 = ((1 - 2*(np.subtract.outer(y_x,tx).T)*l_px[:,np.newaxis])*
                         l_i_y_x**2)
                h_j_1 = ((1 - 2*(np.subtract.outer(y_y,ty).T)*l_py[:,np.newaxis])*
                         l_i_y_y**2)
                h_k_1 = ((1 - 2*(np.subtract.outer(y_z,tz).T)*l_pz[:,np.newaxis])*
                         l_i_y_z**2)
                h_i_2 = (np.subtract.outer(y_x,tx).T)*l_i_y_x**2
                h_j_2 = (np.subtract.outer(y_y,ty).T)*l_i_y_y**2
                h_k_2 = (np.subtract.outer(y_z,tz).T)*l_i_y_z**2
                M_111 = np.ravel(np.einsum('m,im,jm,km->ijk',
                                           panel_weights,h_i_1,h_j_1,h_k_1))
                M_112 = np.ravel(np.einsum('m,im,jm,km->ijk',
                                           panel_weights,h_i_1,h_j_1,h_k_2))
                M_121 = np.ravel(np.einsum('m,im,jm,km->ijk',
                                           panel_weights,h_i_1,h_j_2,h_k_1))
                M_211 = np.ravel(np.einsum('m,im,jm,km->ijk',
                                           panel_weights,h_i_2,h_j_1,h_k_1))
                M_122 = np.ravel(np.einsum('m,im,jm,km->ijk',
                                           panel_weights,h_i_1,h_j_2,h_k_2))
                M_212 = np.ravel(np.einsum('m,im,jm,km->ijk',
                                           panel_weights,h_i_2,h_j_1,h_k_2))
                M_221 = np.ravel(np.einsum('m,im,jm,km->ijk',
                                           panel_weights,h_i_2,h_j_2,h_k_1))
                M_222 = np.ravel(np.einsum('m,im,jm,km->ijk',
                                           panel_weights,h_i_2,h_j_2,h_k_2))       
                panel.far_field_sum = np.array([M_111, M_112, M_121, M_211,
                                                M_122, M_212, M_221, M_222])           
        current_level += 1
   
    tree_time = time.time() - tree_time_start    
            
    #------------------YOU CAN LOOP HERE OVER MAC------------------------------ 

    ###Saving the tree to file (temporarily) for each process to use
    tree['MAC'] = MAC #to load into each process during multi-P
    tree['method'] = method #to load into each process during multi-P
    tree['node_num'] = p
    with open('tree_file.pickle', 'wb') as handle:
       pickle.dump(tree, handle, protocol = pickle.HIGHEST_PROTOCOL)

    ###Multiprocessing initialization
    print('initializing multiprocessing: compute interactions')
    usable_cores = mp.cpu_count()-2 #adjustable
    indices = np.arange(num_points)
    chunks = np.array_split(indices, usable_cores) # splits indices into chunks
    start_time = time.time()           
    pool = mp.Pool(usable_cores) # creates pool of processes
    out = pool.map(chunker,chunks) # maps indices to function in each process
    pool.close()
    pool.join()
    print('processes completed and joined')       
    result = np.concatenate(out)        
    time_taken = time.time() - start_time + tree_time
    
    #This tests a random selection of points to approximate global error
    print('initializing multiprocessing: testing error')
    num_test = usable_cores * 250 #you can adjust this parameter to test more points
    test_ratio = num_points / num_test
    test_indices = np.random.choice(points.shape[1], num_test, replace=False) 
    test_chunks = np.array_split(test_indices, usable_cores)   
    start = time.time()
    pool = mp.Pool(usable_cores)
    out = pool.map(direct_chunker,test_chunks)
    pool.close()
    pool.join()
    print('processes completed and joined')          
    exact = np.concatenate(out) 
    direct_time_est = (time.time()-start) * test_ratio
    error = np.sqrt(np.sum(((result[test_indices] - exact)**2)/
                            np.sum(exact**2)))
         
    print('Treecode time taken:',time_taken,'seconds')
    print('estimated direct sum time:',direct_time_est,'seconds')
    print('estimated error:',error)
        
    ###TO BE DONE ONCE AT THE END OF THE PROGRAM TO CLEAN UP TEMP FILES
    os.remove('tree_file.pickle') # deletes tree structure from directory
    os.remove('data_3D.npy')      # comment this out to save data
    os.remove('weights_3D.npy')   # comment this out to save weights
