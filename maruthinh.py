'''
This code is a modified version of Maruthi N. Hanumantharayappa's code available at:
https://github.com/maruthinh/d2q9_zero_for_loop
'''
import os
import glob
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sp
import torch

from utils import definegetch, get_latest_file
getch = definegetch()

class D2Q9:

        def __init__(self,qmodel, T0):
            self.c = np.array([[0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0],
                               [0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0]])
            self.w = np.array([4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
                               1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0])
            self.q = 9
            self.T0 = T0
            self.a = np.sqrt(self.T0)
            self.qmodel = qmodel

        def feq(self, moments):
            """
            To compute equilibrium distribution function.
            """
            
            feq = np.zeros((self.q, moments.shape[1], moments.shape[2]))
                
            udotc = (np.array([[self.c[0]]]).T*moments[1]) + (np.array([[self.c[1]]]).T*moments[2])
            usq = moments[1] * moments[1] + moments[2] * moments[2]
            feq = (np.array([[self.w]]).T*moments[0]) * (1.0 + 3.0 * udotc 
            - 1.5 * usq + 4.5 * udotc ** 2
            # + 4.5 * udotc ** 3 - 4.5 * usq * udotc
            )
            
            return feq
            
        def moments(self, f):
            """
            Compute moments (rho, u, v) from populations 
            """
            mom = np.zeros((3, f.shape[1], f.shape[2]))

            mom[0] = f.sum(axis=0)
            mom[1] = (np.array([[self.c[0]]]).T*f).sum(axis=0)/mom[0]
            mom[2] = (np.array([[self.c[1]]]).T*f).sum(axis=0)/mom[0]

            return mom

        @torch.no_grad
        def collision(self, f, beta, cutoff, tot_nx,tot_ny, alpha=2.0):
            """
            computes collision term 
            """
            feq = self.feq(self.moments(f))
                
            # Perform quantum collision using the selected quantum neural network
            # Write the mapping between the 9 discrete densities and the 16 values the circuit requires as an input
                
            # Flatten the array of discrete densities over all lattice sites
            fpre = np.swapaxes(f,0,2).reshape((int(tot_ny*tot_nx),9))
            # Perform the quantum collision and return the values
            with torch.no_grad():
                fpost = self.qmodel(torch.tensor(fpre, device = 'cpu'))

            # Reshape the array into its original shape
            fpost = np.array(fpost)
        
            fpost = np.swapaxes(fpost.reshape((tot_ny,tot_nx,9)),2,0)
            arrayCheck = np.greater(np.sqrt((self.moments(f)[1]**2+self.moments(f)[2]**2)),cutoff)    
                           
            modfeq = (1-arrayCheck)*feq+arrayCheck*(fpost)
            f += -alpha*beta*(f-modfeq)
            
            f.cumsum(axis = 0)          
                    
        @staticmethod
        def advection(f):
            """
            Streaming of velocities
            """

            f[3, :-1, :] = f[3, 1:, :]
            f[4, :, :-1] = f[4, :, 1:]
            f[7, :-1, :-1] = f[7, 1:, 1:]
            f[8, 1:, :-1] = f[8, :-1, 1:]

            f[1, -1:0:-1, ::-1] = f[1, -2::-1, ::-1]
            f[2, ::-1, -1:0:-1] = f[2, ::-1, -2::-1]
            f[5, -1:0:-1, -1:0:-1] = f[5, -2::-1, -2::-1]
            f[6, -2:0:-1, -1:0:-1] = f[6, -1:1:-1, -2::-1]

class SimParams(D2Q9):
    def __init__(self, mach, reynolds_num, ref_len, res, qmodel, T0):
        self.mach_num = mach
        self.reynolds = reynolds_num
        self.ref_len = ref_len
        self.res = res
        self.dt = self.res
        D2Q9.__init__(self, qmodel, T0)

    @property
    def ref_vel(self):
        """
        Compute reference velocity
        """
        return self.mach_num * self.a

    @property
    def knudsen(self):
        """
        Compute Knudsen number
        """
        return self.mach_num / self.reynolds

    @property
    def nu(self):
        """
        Compute viscosity from reference length, velocity and Reynolds number
        """
        return self.ref_len * self.ref_vel / self.reynolds

    @property
    def tau0(self):
        return self.nu / self.T0

    @property
    def tau_ndim(self):
        return self.tau0 / self.dt

    @property
    def beta(self):
        """
        Relaxation 
        """
        return 1.0 / (2.0 * self.tau_ndim + 1.0)

class BoundaryConditions(D2Q9):

    def __init__(self, f, qmodel,T0):
        self.f = f
        self.f[:, 0, :] = self.f[:, 1, :]
        self.f[:, -1, :] = self.f[:, -2, :]
        self.f[:, :, 0] = self.f[:, :, 1]
        self.f[:, :, -1] = self.f[:, :, -2]
        D2Q9.__init__(self,qmodel,T0)
        
    def periodic_in_x(self):
        """
        To apply periodic boundary condition in x
        """
        self.f[1, 0, 1:-1] = self.f[1, -1, 1:-1]
        self.f[8, 0, 1:-1] = self.f[8, -1, 1:-1]
        self.f[5, 0, 1:-1] = self.f[5, -1, 1:-1]

        self.f[6, -1, 1:-1] = self.f[6, 0, 1:-1]
        self.f[3, -1, 1:-1] = self.f[3, 0, 1:-1]
        self.f[7, -1, 1:-1] = self.f[7, 0, 1:-1]

    def bounce_back_right(self):
        """
        Apply bounce back on right wall
        """
        self.f[3, -2, 2:-2] = self.f[1, -1, 2:-2]
        self.f[6, -2, 2:-2] = self.f[8, -1, 2:-2]
        self.f[7, -2, 2:-2] = self.f[5, -1, 2:-2]

    def bounce_back_left(self):
        """
        Apply bounce back on right wall
        """
        self.f[8, 1, 2:-2] = self.f[6, 0, 2:-2]
        self.f[1, 1, 2:-2] = self.f[3, 0, 2:-2]
        self.f[5, 1, 2:-2] = self.f[7, 0, 2:-2]

    def bounce_back_bottom(self):
        """
        Apply bounce back on right wall
        """
        self.f[2, 2:-2, 1] = self.f[4, 2:-2, 0]
        self.f[6, 2:-2, 1] = self.f[8, 2:-2, 0]
        self.f[5, 2:-2, 1] = self.f[7, 2:-2, 0]

    def moving_wall_bc_top(self, u_wall):
        """
        Apply moving wall bounce back on top wall
        """
        density=self.f.sum(axis=0)
        self.f[4, 1:-1, -2] = self.f[2, 1:-1, -1]
        self.f[7, 1:-1, -2] = self.f[5, 1:-1, -1] + 6.0 * density[1:-1, -1] * self.w[7] * self.c[0][7] * u_wall
        self.f[8, 1:-1, -2] = self.f[6, 1:-1, -1] + 6.0 * density[1:-1, -1] * self.w[8] * self.c[0][8] * u_wall

    def bot_left_corner_correction(self):
        """
        Bottom left corner
        """
        self.f[2, 1, 1] = self.f[4, 1, 0]
        self.f[5, 1, 1] = self.f[7, 1, 0]
        self.f[1, 1, 1] = self.f[3, 1, 0]
        self.f[8, 1, 1] = self.f[6, 1, 0]
        self.f[6, 1, 1] = self.f[8, 1, 0]

    def bot_right_corner_correction(self):
        """
        Bottom right corner
        """
        self.f[2, -2, 1] = self.f[4, -2, 0]
        self.f[6, -2, 1] = self.f[8, -2, 0]
        self.f[3, -2, 1] = self.f[1, -2, 0]
        self.f[5, -2, 1] = self.f[7, -2, 0]
        self.f[7, -2, 1] = self.f[5, -2, 0]

    def top_left_corner_correction(self):
        """
        Top left corner
        """
        self.f[5, 1, -2] = self.f[7, 1, -1]
        self.f[7, 1, -2] = self.f[5, 1, -1]
        self.f[1, 1, -2] = self.f[3, 1, -1]
        self.f[8, 1, -2] = self.f[6, 1, -1]
        self.f[4, 1, -2] = self.f[2, 1, -1]

def mainLoop(qmodel,lattice_size,ref_len,Mach,Re, T0, alpha, cutoff, min_err,max_iter,output_every,plot_every, save_every):

    folder_string = 'Ma-'+str(Mach)+'-Re-'+str(Re)+'-a-'+str(alpha)+'-'+str(lattice_size)+'x'+str(lattice_size)
    if not cutoff == 0:
        folder_string += '-cutoff-'+str(cutoff)
    # Create the subfolder if it doesn't exist
    if not os.path.exists(folder_string):
        os.makedirs(folder_string)

    lattice_number = int(np.log2(lattice_size))
    initialized = False
    for latnum in range(lattice_number):
        itr_string = 'Ma-'+str(Mach)+'-Re-'+str(Re)+'-a-'+str(alpha)+'-'+str(int(lattice_size/2**latnum))+'x'+str(int(lattice_size/2**latnum))
        if os.path.exists(itr_string):
            print("Folder found")
            file = get_latest_file('./'+itr_string,'npz')
            if file is None:
                print("No file for lattice size "+str(int(lattice_size/2**latnum))+"x"+str(int(lattice_size/2**latnum))+" has been found")
                if latnum != lattice_number-1:
                    print("Looking for coarser grid data")
                continue
            else:
                initialdata = np.load(file)
                initialized = True
                print("Initial data has been found")
                break
    
    nx = lattice_size
    ny = nx
    ghost_x = 2
    ghost_y = 2
    tot_nx = nx + ghost_x
    tot_ny = ny + ghost_y
    ndim = 2
    
    res=ref_len/nx
       
    model = D2Q9(qmodel, T0)
    sim_params = SimParams(mach=Mach, reynolds_num=Re, ref_len=ref_len, res=res, qmodel = qmodel, T0 = T0)
    q = model.q
    
    rho_ini = 1
    p = rho_ini * model.T0
    u_ini = 0.
    v_ini = 0.

    #initialization of arrays required
    f = np.zeros((q, tot_nx, tot_ny), dtype = np.double)
    moments = np.zeros((3, tot_nx, tot_ny))

    u0 = np.zeros((tot_nx, tot_ny))
    v0 = np.zeros((tot_nx, tot_ny))
    f0 = 1.*f

    # to plot contours of velocity
    x = np.linspace(0, ref_len, nx)
    y = np.linspace(0, ref_len, ny)
    X, Y = np.meshgrid(x, y)

    # to interpolate data
    xfine = np.linspace(0, ref_len, tot_nx)
    yfine = np.linspace(0, ref_len, tot_ny)
    Xfine, Yfine = np.meshgrid(xfine, yfine)
    
    # check if grid or coarser grid data exists
    coarse_size = 0
    if initialized == True: 
        coarse_size = int(lattice_size/2**latnum)
        
        iter = file[file.rfind('-')+1:file.rfind('.')]
        iter = int(iter)

        # optimize output frequency
        previous_ratio = iter/len(initialdata['error'])- (iter%len(initialdata['error']))
        
        new_output_every = int(max(10**(int(np.log10(previous_ratio))+1),10))
        new_plot_every = int(max(10**(int(np.log10(iter))-2),100))
        print("Output and plot update frequency has been updated from "+str(output_every)+" and "+str(plot_every)+" to "+str(new_output_every)+" and "+str(new_plot_every)+" respectively")

        output_every = new_output_every
        plot_every = new_plot_every
        save_every = plot_every
        
        err = initialdata['error'][-1]
        l1err = initialdata['l1'][-1]
        
        if latnum != lattice_size:
            # Coarse grid and data
            xcoarse = np.linspace(0,ref_len,coarse_size+ghost_x)
            ycoarse = np.linspace(0,ref_len,coarse_size+ghost_y)
    
            # Create an interpolator object
            interp = sp.interpolate.RegularGridInterpolator((ycoarse, xcoarse), np.swapaxes(initialdata['f'],0,2))
                    
            # Generate the points on the finer grid
            points = np.stack([Yfine.ravel(),Xfine.ravel()], axis=-1)
            
            # Interpolate the data to the finer grid
            f = np.swapaxes(interp(points).reshape((tot_ny,tot_nx,model.q)),2,0)
            
            print("Initial data has been interpolated from grid of size "+str(coarse_size)+'x'+str(coarse_size)+" at iteration "+str(iter))
        else:
            f = initialdata['f']
            
        moments = model.moments(f)
        u0 = 1.*moments[1]
        v0 = 1.*moments[2]
        f0 = 1.*f
        
    
    else:
        moments[0] = rho_ini
        moments[1] = u_ini
        moments[2] = v_ini
        f = model.feq(moments)
        iter = 0
        err = 1.0
        l1err = 1.0
        print("Field has been initilized with zero velocity and unity density")
    
    perr = 1.0
    max_err = 1.0
    
    start = time.time()

    print("Mach: "+str(Mach))
    print("Reynolds "+str(Re))
    print("Lattice Size: "+str(lattice_size))
    print("alpha: "+str(alpha))
    print("beta: "+str(sim_params.beta))
    print("res: "+str(sim_params.res))
    print("ref_vel:"+str(sim_params.ref_vel))
    
    if initialized == False or (initialized == True and coarse_size != lattice_size):
        print("Existing files in respective folder would be overwritten")
    
    print("Press any key to continue...")
    getch()
    print("Continuing...")
        
    if initialized == False or (initialized == True and coarse_size != lattice_size):
        try:
            os.system("rm ./"+folder_string+"/*.png")
            os.system("rm ./"+folder_string+"/*.npz")
    
        except:
            print("No files to remove")
        
    error_array = []
    l1_array = []
    perr_array = []
    #main iteration loop
    # while(err>min_err and iter < max_iter):
    while(max_err>min_err and iter < max_iter):
        iter+=1
        
        bc = BoundaryConditions(f = f,qmodel = qmodel,T0 = T0)
        model.collision(f, sim_params.beta, cutoff,tot_nx,tot_ny)
        model.advection(f)
        # bc.periodic_in_x()
        bc.bounce_back_left()
        bc.bounce_back_right()
        bc.bounce_back_bottom()
        bc.moving_wall_bc_top(sim_params.ref_vel)
        
        
        if iter % output_every == 0:
            moments=model.moments(f)
            e1 = np.sum(np.sqrt((moments[1] - u0) ** 2 + (moments[2] - v0) ** 2))
            e2 = np.sum(np.sqrt(moments[1] ** 2 + moments[2] ** 2))
            err = e1 / e2
            # l1err = np.sum(np.abs(f-f0))
            l1err = np.max(np.abs(moments[1]-u0)+np.abs(moments[2]-v0))/sim_params.ref_vel
            perr = np.max(np.abs(moments[0]-np.sum(f0,0)))
            # max_err = np.max(np.array([perr,l1err]))
            max_err = l1err
            print('Relative Velocity Error = ',err,", Max L1 Norm Velocity Error = ",l1err,", Max L1 Density Error = ", perr)
            
            VelNorm = np.sqrt(moments[1, 1:-1, 1:-1] * moments[1, 1:-1, 1:-1] 
                      + moments[2, 1:-1, 1:-1] * moments[2, 1:-1, 1:-1])
            error_array.append(err)
            l1_array.append(l1err)
            
            if iter % plot_every == 0 or (max_err < min_err):
                plt.figure()
                cp = plt.contour(X, Y, np.transpose(VelNorm),15)
                # norm= mpl.colors.Normalize(vmin=cp.cvalues.min(), vmax=cp.cvalues.max())
                norm= mpl.colors.Normalize(0., 1.)
                sm = plt.cm.ScalarMappable(norm=norm, cmap = cp.cmap)
                sm.set_array([])
                # cbar=plt.colorbar(sm,ticks=cp.levels,ax=plt.gca())
                cbar=plt.colorbar(sm,ax=plt.gca())
                cbar.set_label('Velocity', fontsize=12)
                plt.xlabel('x', fontsize=12)
                plt.ylabel('y', fontsize=12)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.savefig(folder_string+'/velocity_contours_'+str(int(iter))+'.png', bbox_inches='tight')
                plt.close()
        
        u0 = 1.*moments[1]
        v0 = 1.*moments[2]
        f0 = 1.*f

        if iter % save_every == 0:
            np.savez(folder_string+'/data-'+str(int(iter)),f = f, u = VelNorm, error = np.array(error_array), l1 = np.array(l1_array), ux = u0, uy = v0, ref_vel = sim_params.ref_vel)
    print("====================")
    print("alpha: "+str(alpha))
    print("beta: "+str(sim_params.beta))
    print("res: "+str(sim_params.res))
    print("ref_vel"+str(sim_params.ref_vel))
    print('error is = ', err)
    print('L1 error is = ', l1err)
    print('perr is = ', perr)
    np.savez(folder_string+'/data-'+str(int(iter)),f = f, u = VelNorm, error = np.array(error_array), l1 = np.array(l1_array), ux = u0, uy = v0, ref_vel = sim_params.ref_vel)
    print("total number of iterations:"+str(iter))
    print("total time taken: ", datetime.timedelta(seconds=(time.time()-start)))
