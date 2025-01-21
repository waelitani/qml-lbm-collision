'''

This code is a modified version of the code published by Alessandro Gabbana at:
https://github.com/agabbana/learning_lbm_collision_operator
to accompany the paper:
Corbetta, A., Gabbana, A., Gyrya, V. et al. Toward learning Lattice Boltzmann collision operators. Eur. Phys. J. E 46, 10 (2023). https://doi.org/10.1140/epje/s10189-023-00267-w

'''
import numpy as np

from corbetta import LB_stencil, compute_rho_u, compute_f_rand, compute_f_pre_f_post, compute_feq, delete_negative_samples

#####################################
# settings 

n_samples = 1_000_000

u_abs_min = 1e-5
u_abs_max = 1
sigma_min = 1e-15 
sigma_max = 5e-2

#####################################
# lattice velocities and weights
Q = 9 
global c
c, w, cs2 = LB_stencil()

#####################################

fPreLst  = np.empty( (n_samples, Q) )
fPostLst = np.empty( (n_samples, Q) )
fEqLst   = np.empty( (n_samples, Q) )
uxLst = np.empty( (n_samples, 1) )
uyLst = np.empty( (n_samples, 1) )
rhoLst = np.empty( (n_samples, 1) )

#####################################

idx = 0

# loop until we get n_samples without negative populations
while idx < n_samples: 
    
    # get random values for macroscopic quantities
    rho, u = compute_rho_u(n_samples)

    rho = rho[:,np.newaxis]
    ux  = u[:,0][:,np.newaxis]
    uy  = u[:,1][:,np.newaxis]

    # compute the equilibrium distribution
    f_eq  = np.zeros((n_samples, 1, Q))
    f_eq  = compute_feq(f_eq, rho, ux, uy, Q)[:,0,:]
    
    # compute a random non equilibrium part
    f_neq = compute_f_rand(n_samples, sigma_min, sigma_max)   
    
    # apply BGK to f_pre = f_eq + f_neq
    tau , f_pre, f_post = compute_f_pre_f_post(f_eq, f_neq)
    
    # remove negative elements
    f_eq, f_pre, f_post = delete_negative_samples(n_samples, f_eq, f_pre, f_post)
    
    # accumulate 
    non_negatives = f_pre.shape[0]
    
    idx1        = min(idx+non_negatives, n_samples)
    to_be_added = min(n_samples-idx, non_negatives)
    
    fPreLst[ idx:idx1] = f_pre[ :to_be_added]
    fPostLst[idx:idx1] = f_post[:to_be_added]
    fEqLst[  idx:idx1] = f_eq[  :to_be_added]
    uxLst[ idx: idx1] = ux[ : to_be_added]
    uyLst[ idx: idx1] = uy[ : to_be_added]
    rhoLst[ idx: idx1] = rho[ : to_be_added]
    
    idx = idx + non_negatives 
    
np.savez('example_dataset.npz', 
        f_pre  = fPreLst,
        f_post = fPostLst,
        f_eq   = fEqLst,
       ux = uxLst,
       uy = uyLst,
       rho = rhoLst
       )

