import os
import glob
import torch
# Import the functions which constructs our quantum neural network
from distributedelastic import load_model, load_snapshot
import re
import numpy as np

def definegetch():
    if os.name == 'nt':
        import msvcrt
        def getch():
            return msvcrt.getch()
    else:
        import sys, tty, termios
        def getch():
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch
    return getch

def checkModel(datapoint,params):
    select_model = False
    if 'layers' in params:
            if params['layers'] == int(datapoint['layers']):
                select_model = True
            else:
                select_model = False
    if 'binary_precision' in params:
        if params['binary_precision'] == int(datapoint['binary_precision']):
            select_model = select_model and True
        else:
            select_model = select_model and False
    if 'ancillas' in params:
        if params['ancillas'] == int(datapoint['ancillas']):
            select_model = select_model and True
        else:
            select_model = select_model and False
    return select_model

@torch.no_grad
def loadModel(model_name,model_args_dict,weights_path = './trained_weights/', lib_path = './circuit_library/'):

    dir_path = weights_path+model_name+'/'
    selected_model = None
    # Create an array to store the log files associated with the trained weights
    logfiles = []
    # Iterate directory
    for file in os.listdir(dir_path):
        # check only text files
        if os.path.isdir(os.path.join(dir_path, file)) and file.count('-') == 3:
            logfiles.append(file)            

    # Store the file names corresponding the quantum circuit parameters
    nbFiles = len(logfiles)
    logstring = []
    namestring = []
    for i in logfiles:
        list_to_add = re.split('-',i)
        list_to_add[-1] = list_to_add[-1]
        logstring.append(list_to_add)
        namestring.append(i)

    # Create a dictionary storing the parameters of each circuit as well as its training history
    datalog = []
    for i in range(nbFiles):
        rawdata = np.genfromtxt(dir_path+logfiles[i]+'/log.out', delimiter=',')
        datalog.append({"epoch":rawdata[:,0],
        "loss":rawdata[:,1],
        "binary_precision":logstring[i][0],
        "layers":logstring[i][1],
        "reps":logstring[i][2],
        "ancillas":logstring[i][3],
        "filename":namestring[i]})
        
        select_model = checkModel(datalog[i],model_args_dict)
        if select_model:
            selected_model = i
    
    print(namestring)
    # Construct the model with the trained weights
    qmodel, _, _, _, _ = load_model(lib_path+model_name, model_args_dict)

    if selected_model is None:
        print('No trained weights have been found for the provided model parameters')
    else:
        try:
            datapoint = datalog[selected_model]
            print('Using the weights in '+namestring[selected_model])
            load_snapshot(qmodel, snapshot_path = dir_path+namestring[selected_model]+'/')
            qmodel.eval()
        except:
            print("Unable to load trained weights")
    return qmodel

def get_latest_file(directory, extension):
    """Gets the latest file in a directory with the given extension."""
    files = glob.glob(directory+'*.'+extension)
    if not files:
        return None
        
    return max(files, key=os.path.getmtime)
