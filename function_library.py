# Import the necessary libraries
import os
import numpy as np
import pennylane as qml

import torch 
torch.set_default_dtype(torch.float64)
torch.set_default_device('cpu')

from corbetta import LB_stencil, load_data

# Retrieve the lattice basis vectors
c, _, _ = LB_stencil()
D = 2
Q = 9
c = torch.tensor(data = c, dtype = torch.float64, device = "cpu")

class Dataset(torch.utils.data.Dataset):
  def __init__(self, X,y):
        'Initialization'
        self.X = X
        self.y = y

  def __len__(self):
        return len(self.X)

  def __getitem__(self, index):

        # Load data and get label
        X = self.X[index]
        y = self.y[index]

        return X, y

def d8transform(qc):
    transform_reference = torch.tensor([[0,1,2,3,4,5,6,7,8],[0,7,8,1,2,3,4,5,6],[0,5,6,7,8,1,2,3,4],[0,3,4,5,6,7,8,1,2],
                    [0,1,8,7,6,5,4,3,2],[0,5,4,3,2,1,8,7,6],[0,3,2,1,8,7,6,5,4],[0,7,6,5,4,3,2,1,8]])
    
    unitary = torch.zeros((2**qc,2**qc))
    unitaries = []
    for i in range(8):
        transform_matrix = torch.clone(unitary)
        
        for l in range(Q):
             transform_matrix[transform_reference[i][l],l] = 1
         
        for l in range(Q,2**qc):
            transform_matrix[l,l] = 1
        
        m = transform_matrix
        
        class transform_matrix(qml.operation.Operation):

            num_params = 0
            num_wires = 4
            par_domain = None

            @staticmethod
            def compute_matrix():
                return np.array(m)
        
        unitaries.append(transform_matrix)
        
    return unitaries
    
# Load the sample dataset from the .npz dataset file generated
def dataLoad(qc, num_samples, test_size = 0.1, truncate = False, binary_precision = 1):
    # # Data Load
    # read training dataset
    fpre, fpost = load_data('example_dataset.npz')
    
    fpre = fpre[0:num_samples]
    fpost = fpost[0:num_samples]
    if truncate:
        fpost = np.around(fpost, decimals = int(np.floor(binary_precision*np.log(2)/np.log(10))))
    # Calculate the exact output values in the binary precision used for encoding for a fair comparison
    return Dataset(torch.tensor(fpre,requires_grad = True, device = "cpu"),torch.tensor(fpost,requires_grad = True, device = "cpu"))
    
# Calculate the momentum
def calculate_u(f,c):
    u = torch.matmul(f,c)
    return (u)

# Evaluate the loss function as root mean square error of the discrete densities as well as the velocity and fluid density
def massmomentumloss(target, output):
    utarget = calculate_u(target,c)
    uoutput = calculate_u(output,c)
    rhotarget = torch.sum(output, dim = 1)
    rhooutput = torch.sum(target, dim = 1)
    loss = (1./3.)*(torch.mean(torch.square(output - target))+torch.mean(torch.square(uoutput-utarget))+torch.mean(torch.square(rhooutput-rhotarget)))
    return torch.sqrt(loss)
    
def float_bin(number,places):
    #only the decimal portion is converted
    number = number-int(torch.floor(number))
    
    s = ''
    for p in range(places):
        weight = 1./(2.**(p+1))
        res = number-weight
        if res>=0:
            s = s+'1'
            number = number-weight
        else:
            s = s+'0'
    return s
    
def binaryAmplitude(f, qc = 4, binary_precision = 8):
    num_samples = int(f.shape[0])
    num_vars = int(f.shape[1])
    bin_prec = int(binary_precision)

    outset = torch.zeros(size = (num_samples,2**(qc+bin_prec)), device = 'cpu')

    for s in range(num_samples):
        for q in range(num_vars):
            id = int(np.binary_repr(q,qc)+float_bin(f[s][q],bin_prec),2)
            outset[s][id] = 1

    return outset