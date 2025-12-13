from mpi4py import MPI
import time

class MasterNode:
    def __init__(self, comm, size):
        """
        Initialize the Master Node.
        :param comm: The MPI communicator (allows talking to other nodes)
        :param size: Total number of processes (Master + Workers)
        """
        self.comm = comm
        self.num_workers = size - 1  
        self.job_queue = list(range(20))
        
        print(f"[Master] System initialized with {self.num_workers} workers and {len(self.job_queue)} batches.")
    
    def run_dynamic_scheduler(self):
        """
        The main loop:
        1. Wait for a request (READY or DONE).
        2. If work exists, send it.
        3. If no work exists, send STOP.
        """
        active_workers = self.num_workers
        
        print("[Master] Scheduler started. Waiting for requests...")

        start_time = time.time()
        
        while active_workers > 0:
            status = MPI.Status()
            
            message = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            
            worker_rank = status.Get_source()
            msg_type = message.get('status')

            if self.job_queue:
                batch_id = self.job_queue.pop(0)
                
                self.comm.send({'batch_id': batch_id}, dest=worker_rank, tag=100)
                
                print(f"[Master] Sent Batch {batch_id} -> Worker {worker_rank}")
            
            else:
                self.comm.send({'batch_id': None}, dest=worker_rank, tag=100)
                
                active_workers -= 1
                print(f"[Master] Job Queue empty. Sent STOP -> Worker {worker_rank}")

        end_time = time.time()
        print(f"[Master] Dynamic Scheduling Completed in {end_time - start_time:.2f} seconds.") # <--- ADD THIS
        print("[Master] All workers finished. Shutting down.")
        
    def run_static_scheduler(self):
        """
        STATIC SCHEDULING (The 'Control Group')
        1. Pre-calculate the split (e.g., 20 batches / 2 workers = 10 each).
        2. Assign specific batches to specific workers.
        3. No stealing allowed! If a worker is slow, everyone waits.
        """
        print("[Master] Starting STATIC scheduling (Equal Split).")
        
        start_time = time.time()

        # 1. Divide work equally
        batches_per_worker = len(self.job_queue) // self.num_workers
        remainder = len(self.job_queue) % self.num_workers
        
        current_batch_index = 0
        
        # 2. Assign batches upfront
        for worker_rank in range(1, self.num_workers + 1):
            count = batches_per_worker + (1 if worker_rank <= remainder else 0)
            assigned_batches = self.job_queue[current_batch_index : current_batch_index + count]
            current_batch_index += count
            
            print(f"[Master] Pre-assigning {len(assigned_batches)} batches to Worker {worker_rank}")
            
            for batch_id in assigned_batches:
                status = MPI.Status()
                self.comm.recv(source=worker_rank, tag=MPI.ANY_TAG, status=status)
                
                self.comm.send({'batch_id': batch_id}, dest=worker_rank, tag=100)
                print(f"[Master] Static Logic: Forced Batch {batch_id} -> Worker {worker_rank}")

            # 3. Send STOP signal to this specific worker when they finish their quota
            self.comm.recv(source=worker_rank, tag=MPI.ANY_TAG, status=status) # Final DONE check
            self.comm.send({'batch_id': None}, dest=worker_rank, tag=100)
            print(f"[Master] Worker {worker_rank} finished their static assignment.")
        
        end_time = time.time()
        print(f"[Master] Static Schedule Completed in {end_time - start_time:.2f} seconds.")
        print("[Master] Static Schedule Completed.")

    def start(self):
        self.run_dynamic_scheduler()