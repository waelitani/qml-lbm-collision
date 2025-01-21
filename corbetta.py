'''

This code is an adaptation of the code published by Alessandro Gabbana at:
https://github.com/agabbana/learning_lbm_collision_operator
to accompany the paper:
Corbetta, A., Gabbana, A., Gyrya, V. et al. Toward learning Lattice Boltzmann collision operators. Eur. Phys. J. E 46, 10 (2023). https://doi.org/10.1140/epje/s10189-023-00267-w

'''

# Import the necessary libraries
import numpy as np
from numba import jit

# Load the pre- and post- collision discrete density sample dataset from the .npz file
def load_data(fname):

    data = np.load(fname, allow_pickle=True)

    fpre  = data['f_pre']
    fpost = data['f_post']
    
    return fpre, fpost    

# Generate random velocity and density values within the given bounds
def compute_rho_u(num_samples, rho_min=0.95, rho_max=1.05, u_abs_min=0.0, u_abs_max=0.01):
    
    rho   = np.random.uniform(rho_min, rho_max, size=num_samples)    
    u_abs = np.random.uniform(u_abs_min, u_abs_max, size=num_samples)
    theta = np.random.uniform(0, 2*np.pi, size=num_samples)
    
    ux = u_abs*np.cos(theta)
    uy = u_abs*np.sin(theta)
    u  = np.array([ux,uy]).transpose()
    
    return rho, u

def compute_f_rand(num_samples, sigma_min, sigma_max):

    Q  = 9
    K0 = 1/9.
    K1 = 1/6.

    c, w, cs2 = LB_stencil()

    #########################################
    
    f_rand = np.zeros((num_samples, Q))

    #########################################
    
    if sigma_min==sigma_max:
        sigma = sigma_min*np.ones(num_samples)
    else:
        sigma = np.random.uniform(sigma_min, sigma_max, size=num_samples)    

    #########################################        
        
    for i in range(num_samples):
        f_rand[i,:] = np.random.normal(0, sigma[i], size=(1,Q))

        rho_hat = np.sum(f_rand[i,:]       )
        ux_hat  = np.sum(f_rand[i,:]*c[:,0])
        uy_hat  = np.sum(f_rand[i,:]*c[:,1])

        f_rand[i,:] = f_rand[i,:] -K0*rho_hat -K1*ux_hat*c[:,0] -K1*uy_hat*c[:,1]  

    return f_rand


def compute_f_pre_f_post(f_eq, f_neq, tau_min=0.95, tau_max=1.05):
    tau    = np.random.uniform(tau_min, tau_max, size=f_eq.shape[0])
    
    f_pre  = f_eq + f_neq
    
    f_post = f_pre + 1/tau[:,None]*(f_eq - f_pre)

    return tau, f_pre, f_post


def delete_negative_samples(n_samples, f_eq, f_pre, f_post):
    
    i_neg_f_eq   = np.where(np.sum(f_eq  <0,axis=1) > 0)[0]
    i_neg_f_pre  = np.where(np.sum(f_pre <0,axis=1) > 0)[0]
    i_neg_f_post = np.where(np.sum(f_post<0,axis=1) > 0)[0]

    i_neg_f = np.concatenate( (i_neg_f_pre, i_neg_f_post, i_neg_f_eq) )
    
    f_eq   = np.delete(np.copy(f_eq)  , i_neg_f, 0)
    f_pre  = np.delete(np.copy(f_pre) , i_neg_f, 0)
    f_post = np.delete(np.copy(f_post), i_neg_f, 0)
    
    return f_eq, f_pre, f_post

@jit
def LB_stencil():
    
    ###########################################################
    # D2Q9 stencil 
    Q = 9
    c = np.zeros((Q, 2), dtype=np.float64)
    w = np.zeros(Q, dtype=np.float64)    
            
    cs2     = 1./3.
    qorder  = 2

    c[0, 0] =  0;  c[0, 1] =  0; w[0] = 4./9.
    c[1, 0] =  1;  c[1, 1] =  0; w[1] = 1./9.
    c[2, 0] =  0;  c[2, 1] =  1; w[2] = 1./9.
    c[3, 0] = -1;  c[3, 1] =  0; w[3] = 1./9.
    c[4, 0] =  0;  c[4, 1] = -1; w[4] = 1./9.
    c[5, 0] =  1;  c[5, 1] =  1; w[5] = 1./36.
    c[6, 0] = -1;  c[6, 1] =  1; w[6] = 1./36.
    c[7, 0] = -1;  c[7, 1] = -1; w[7] = 1./36.
    c[8, 0] =  1;  c[8, 1] = -1; w[8] = 1./36.

    return c, w, cs2

# Function for the calculation of the equilibrium
@jit
def compute_feq(feq, rho, ux, uy, Q):
    c, w, cs2 = LB_stencil()
    uu = (ux**2 + uy**2)*(1./cs2)

    for ip in range(Q):

        cu = (c[ip, 0]*ux[:,:]  + c[ip, 1]*uy[:,:] )*(1./cs2)

        feq[:, :, ip] = w[ip]*rho*(1.0 +cu+0.5*(cu*cu - uu))

    return feq