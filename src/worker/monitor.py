import pynvml
import random

class GPUMonitor:
    def __init__(self, use_simulation=False):
        self.use_simulation = use_simulation
        self.handle = None
        
        if not self.use_simulation:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Monitor GPU 0
                print("[Monitor] NVML Initialized successfully.")
            except Exception as e:
                print(f"[Monitor] NVML failed to init (using simulation): {e}")
                self.use_simulation = True

    def get_metrics(self):
        """
        Returns a dictionary containing:
        - utilization: GPU usage %
        - memory: Memory usage %
        - temperature: GPU Temp (C)
        """
        if self.use_simulation:
            # Simulate fluctuating load
            return {
                'utilization': random.randint(20, 90),
                'temperature': random.randint(40, 75),
                'memory_used': random.randint(1000, 4000)
            }
        
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            
            return {
                'utilization': util.gpu,
                'temperature': temp,
                'memory_used': mem.used / 1024 / 1024 # Convert to MB
            }
        except Exception as e:
            print(f"[Monitor] Error reading metrics: {e}")
            return {'utilization': 0, 'temperature': 0, 'memory_used': 0}