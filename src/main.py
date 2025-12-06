from mpi4py import MPI
import sys
import os

# Add 'src' to python path to find modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.master.scheduler import MasterNode
from src.worker.trainer import WorkerNode

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        app = MasterNode(comm, size)
        app.start()
    else:
        app = WorkerNode(comm, rank)
        app.start()

if __name__ == "__main__":
    main()