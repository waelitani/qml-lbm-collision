import os
import numpy as np
np.int = np.int64
import functools
import builtins
from datetime import datetime

import glob 

import torch
from torch.distributed.elastic.multiprocessing.errors import record

from modularLayer import QNN

torch.set_default_device('cpu')
torch.set_default_dtype(torch.float64)

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
torch.set_grad_enabled(True)

from torchquickstart import train
from corbetta import LB_stencil

from function_library import massmomentumloss,dataLoad

world_size = 0

if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])

distributed = world_size > 1

def load_model(file_path, circuit_scope = {}):
    with open(file_path, 'r') as file:
        file_content = file.read()
        # file should construct an object named qnn
    file_content = "from modularLayer import QNN \n"+file_content
    exec(file_content, circuit_scope)
    qnn = circuit_scope['qnn']

    model, circuit_instructions, circuit_construction, md5identifier = qnn.compileModel()
    return model, circuit_instructions, circuit_construction, md5identifier, qnn
    
def copy_file(source_file, destination_folder):

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    destination_file = os.path.join(destination_folder, os.path.basename(source_file))
    with open(source_file, 'rb') as src, open(destination_file, 'wb') as dst:
        dst.write(src.read())

def setup(rank: int, world_size: int):
    # Initialize the process group
    torch.distributed.init_process_group(backend="gloo", world_size=world_size, rank=rank)

def cleanup():
    "Cleans up the distributed environment"
    torch.distributed.destroy_process_group()
    

def save_snapshot(model, epoch,snapshot_path,filename):
    snapshot = {}
    snapshot["MODEL_STATE"] = model.state_dict()
    snapshot["EPOCHS_RUN"] = epoch
    torch.save(snapshot, snapshot_path+filename+"-"+str(epoch)+".pt")

def load_snapshot(model, snapshot_path = "0"):
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        with open(snapshot_path+"log.out", "a") as thisfile:
                thisfile.write(f"Epoch, Loss")
        epochs_run = 0
    try:
        print('Loading latest weights from '+snapshot_path)
        latest = max(glob.iglob(snapshot_path+'/*.pt'), key=os.path.getmtime)
        snapshot = torch.load(latest
        ,  map_location=torch.device('cpu')
        , weights_only = True)
        model.load_state_dict(snapshot["MODEL_STATE"], strict = False)
        epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Loading weights from snapshot at Epoch {epochs_run}")
    
    except:
        print(f"Creating a new model")
        epochs_run = 0
    return epochs_run

def print_pass(*args):
    pass

@record
def main(num_epochs, save_every, num_samples, batch_size, multiplier, learning_rate,file_path):
    # batch size must be an exact divisor of the number of samples
    args = locals()
    torch.set_default_dtype(torch.float64)

    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        if local_rank != -1:
            rank = local_rank
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            rank = int(os.environ['SLURM_PROCID'])
    else:
        rank = 0 
    
    if rank!=0:
        builtins.print = print_pass
        
    print("Distributed:"+str(distributed))
    
    # model, circuit_instructions, circuit_construction, md5identifier, _ = load_model(file_path)
    model, circuit_instructions, circuit_construction, md5identifier, _ = load_model(file_path, circuit_scope = {'binary_precision' : 2, 'layers' : 4})
        
    str_name = md5identifier
    checkpoint_dir = "./"+str_name+"/"
    
    if rank == 0:
        copy_file(file_path, checkpoint_dir)
    
        with open(checkpoint_dir+"log.out", "a") as thisfile:
            thisfile.write("\n\nMD5: "+md5identifier+" \n")
            for key in args:    
                thisfile.write(str(key)+': '+str(args[key])+" \n")
    
    # # Adjustable Parameters
    # The number of dimensions and discrete velocities is fixed in the data generation algorithm for now
    dimensions = 2
    Q = 3**dimensions
    c, _, _ = LB_stencil()
    
    qc = int(np.ceil(np.log2(Q)))
    
    loss_fn = massmomentumloss  
    optimizer = torch.optim.Adam(params = model.parameters(), lr = learning_rate)
    epochs_run = load_snapshot(model, checkpoint_dir)
    
    trainset = dataLoad(qc, num_samples, test_size = 0)
    
    if distributed:
        setup(rank, world_size)
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle = True)
    else:
        train_sampler = torch.utils.data.RandomSampler(trainset, replacement = False, num_samples = multiplier*batch_size)
    
    train_loader = torch.utils.data.DataLoader(
    trainset
    , shuffle= False
    , sampler= train_sampler
    , batch_size = batch_size
    , pin_memory = False
    )
    
    optimizer.zero_grad()

    # # Training Run

    c = torch.tensor(c, device = 'cpu')
    
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)
    model.train()
       
    min_loss = 2
    epochTimes = []
    loss = []
    
    for t in range(epochs_run, num_epochs):
        print(f"\nEpoch {t+1}\n-------------------------------")
        if distributed:
            train_sampler.set_epoch(t)
        startTime = datetime.now()
        
        current_loss = train(train_loader, model, loss_fn, optimizer, c, rank, world_size)
        runtime = datetime.now()-startTime
        epochs_run += 1
        
        print(runtime)
        
        if rank == 0:
            if t%save_every == 0 and current_loss<min_loss:
                save_snapshot(model, epochs_run,checkpoint_dir,'epoch')
                min_loss = 1.*current_loss
            loss.append(current_loss)
            processed_samples = np.abs(batch_size*multiplier*(t-epochs_run))
            print(f"{current_loss:>9f}")
            print(f"[{processed_samples:>5d}/{num_samples:>5d}]\n")
            with open(checkpoint_dir+"log.out", "a") as thisfile:
                thisfile.write(f"\n{epochs_run+1}, {current_loss}")
        
        epochTimes.append(runtime)
        
    print("Done!")
    
    if distributed:
        cleanup()
    
    pass

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ne', help = "Total number of epochs to run the training", type = int, default = 200)
    parser.add_argument('--se', help = "Frequency of evaluating whether weights have improved to be saved", type = int, default = 1)
    parser.add_argument('--ns', help = "Total number of samples used", type = int, default = 10)
    parser.add_argument('--bs', help = "Batch size", type = int, default = 2)
    parser.add_argument('--mlt', help = "Number of samples drawn for every epoch", type = int, default = 2**0)
    parser.add_argument('--lr', help = "Learning rate", type = float, default = 0.1)
    parser.add_argument('--fl', help = "Name of the file containing the circuit construction code", default = None)
    args = parser.parse_args()
    
    main(args.ne,args.se,args.ns,args.bs,args.mlt,args.lr,args.fl)
