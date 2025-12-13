from mpi4py import MPI
import time
import torch
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.shared.model import SimpleCNN
from src.shared.data_loader import get_data_loader

class WorkerNode:
    def __init__(self, comm, rank):
        self.comm = comm
        self.rank = rank
        self.device = torch.device("cpu")
        
        self.is_straggler = (self.rank % 2 != 0) 
        self.artificial_delay = 1.5 if self.is_straggler else 0.0

        print(f"[Worker {self.rank}] Loading MNIST Dataset and Model...")
        self.model = SimpleCNN().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)
        
        self.data_loader = get_data_loader(batch_size=64)
        self.batches = list(self.data_loader) 

    def train_batch(self, batch_idx):
        """
        Runs the actual Forward + Backward pass on a specific batch of data.
        """
        self.model.train()
        
        safe_idx = batch_idx % len(self.batches)
        data, target = self.batches[safe_idx]
        data, target = data.to(self.device), target.to(self.device)

        self.optimizer.zero_grad()
        output = self.model(data)
        loss = F.nll_loss(output, target)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def start(self):
        print(f"[Worker {self.rank}] Ready to train. Waiting for Master...")
        
        self.comm.send({'status': 'READY'}, dest=0, tag=100)
        
        while True:
            data = self.comm.recv(source=0, tag=100)
            batch_id = data.get('batch_id')
            
            if batch_id is None:
                print(f"[Worker {self.rank}] Received STOP signal.")
                break
            
            print(f"[Worker {self.rank}] Training on Batch {batch_id}...")
            
            # 1. Run actual PyTorch training
            loss = self.train_batch(batch_id)
            
            # 2. Simulate Straggler Effect
            if self.artificial_delay > 0:
                time.sleep(self.artificial_delay)

            print(f"[Worker {self.rank}] Finished Batch {batch_id} (Loss: {loss:.4f})")
            
            self.comm.send({'status': 'DONE'}, dest=0, tag=100)