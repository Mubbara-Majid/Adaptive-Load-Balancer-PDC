from mpi4py import MPI
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.master.scheduler import MasterNode
from src.worker.trainer import WorkerNode

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Default to dynamic if not specified
    mode = "dynamic"
    if len(sys.argv) > 1:
        mode = sys.argv[1]

    if rank == 0:
        # --- MASTER NODE ---
        app = MasterNode(comm, size)
        
        if mode == "static":
            app.run_static_scheduler()
        else:
            app.run_dynamic_scheduler()
            
    else:
        # --- WORKER NODE ---
        # Workers don't need to know the mode; they just obey commands.
        app = WorkerNode(comm, rank)
        app.start()

if __name__ == "__main__":
    main()