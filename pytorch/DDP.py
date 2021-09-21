### Practice of DistributedDataParallel
# ref: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html


import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
 
class ToyModel(nn.Module):
    def __init__(self) -> None:
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.linear(10, 5)
    
    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def demo_basic(rank, world_size):
    print(f'Running basic DDP example on rank {rank}.')
    setup(rank, world_size)
    
    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    loss = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    
    # TODO: more process should be done on Dataloader
    # TODO  for example, DistributedSampler()
    # TODO               pin_memory=True; non_blocking=True; (for faster training)
    outputs = ddp_model(torch.randn(20,10).to(rank))
    
    labels = torch.randn(20, 5).to(rank)
    optimizer.zero_grad()
    loss(outputs, labels).backward()
    optimizer.step()
    
    cleanup()

def run_demo(demo_fn, word_size):
    mp.spawn(
        demo_fn,
        args=(word_size, ),
        nprocs=word_size,
        join=True
    )

def demo_checkpoint(rank, world_size):
    # load model, loss, optimizer ...
    ddp_model = DDP('model', devices_id=[rank])
    
    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        torch.save()
        
    # Use a barrier() to make sure that process 1 loads the model after process
    dist.barrier()
    
    # configure map_location properly, this is essiential
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location))
    
    # train model, compute loss, loss backward
    if rank == 0:
        os.remove(CHECKPOINT_PATH)
        
    cleanup()
    
class ToyMpModel(nn.Module):
    # for large model which need to be initialized at more than one GPU
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)
    
def demo_model_parallel(rank, world_size):
    # when passing a multi-GPU model to DDP, devices_id and output_device must 
    # not be set. Input and output  data will be placed in proper devices by
    # either the application or the model forward() method()
    
    # setup mp_model and devices for this process
    dev0 = (rank * 2) % world_size
    dev1 = (rank * 2 + 1) % world_size
    ddp_model = DDP(ToyMpModel(dev0, dev1))
    
    pass


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run_demo(demo_basic, world_size)
    run_demo(demo_checkpoint, world_size)
    run_demo(demo_model_parallel, world_size)

