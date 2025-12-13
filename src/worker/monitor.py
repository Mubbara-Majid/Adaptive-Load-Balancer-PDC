import pynvml
import random
import time

class GPUMonitor:
    def __init__(self, device_index=0):
        """
        Initializes the GPU Monitor.
        :param device_index: The ID of the GPU to monitor (usually 0).
        """
        self.device_index = device_index
        self.handle = None
        self.simulation_mode = False

        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            gpu_name = pynvml.nvmlDeviceGetName(self.handle)
            print(f"[Monitor] Connected to Real GPU: {gpu_name}")
            
        except pynvml.NVMLError as e:
            print(f"[Monitor] NVML failed to initialize ({e}). Switching to SIMULATION MODE.")
            self.simulation_mode = True
        except Exception as e:
            print(f"[Monitor] Unexpected error: {e}. Switching to SIMULATION MODE.")
            self.simulation_mode = True

    def get_stats(self):
        """
        Returns a dictionary containing the current GPU statistics.
        """
        if self.simulation_mode:
            return self._get_simulated_stats()
        else:
            return self._get_real_stats()

    def _get_real_stats(self):
        try:
            # Get Utilization (GPU % and Memory %)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            gpu_util = utilization.gpu
            
            # Get Memory Info (Used vs Total)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            mem_used_mb = memory_info.used / 1024 / 1024  # Convert Bytes to MB
            
            # Get Temperature (Degrees C)
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            
            return {
                'gpu_util': gpu_util,      # %
                'mem_used': mem_used_mb,   # MB
                'temp': temp,              # Celsius
                'mode': 'REAL'
            }
        except pynvml.NVMLError:
            return self._get_simulated_stats()

    def _get_simulated_stats(self):
        """
        Generates fake data for testing on laptops without GPUs.
        """
        return {
            'gpu_util': random.randint(30, 95),     
            'mem_used': random.randint(1024, 4096),  
            'temp': random.randint(45, 80),          
            'mode': 'SIMULATED'
        }

    def shutdown(self):
        if not self.simulation_mode:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass

# --- TEST BLOCK (Run this file directly to test) ---
if __name__ == "__main__":
    monitor = GPUMonitor()
    print("Reading stats for 5 seconds...")
    for _ in range(5):
        stats = monitor.get_stats()
        print(f"Stats: {stats}")
        time.sleep(1)