import os
import torch
from maruthinh import mainLoop
from utils import loadModel

def main():
    # Set the number of grid points along the side of the square lattice domain
    lattice_size = int(128)
    
    #Simulation parameters
    T0 = 1./3.
    alpha = 2.
    cutoff = 4.3e-3
    Mach= 0.1
    ref_len = 1.
    
    # Re = 2*Mach*lattice_size/np.sqrt(T0)
    # Re = lattice_size**(4/3)
    Re = 100.
    print("Re:"+str(Re))
    
    #Run Parameters
    output_every = 10
    plot_every = 10
    save_every = 10
    min_err = 10**(-12)
    max_iter = 1e7
       
    with torch.no_grad():
        qmodel = loadModel('SEL-CRY-Inverse-SEL',{'layers':64,'binary_precision':1})

    mainLoop(qmodel,lattice_size,ref_len,Mach,Re, T0, alpha, cutoff, min_err,max_iter,output_every,plot_every, save_every)

if __name__ == "__main__":
    main()
    

