# Adaptive Load Balancer for Distributed Deep Learning 🧠⚖️

## Project Overview
This project implements a **Distributed Deep Learning System** using **MPI (Message Passing Interface)**. It demonstrates how **Adaptive (Dynamic) Scheduling** can outperform traditional **Static (Round-Robin)** scheduling in heterogeneous computing environments.

The system trains a **Convolutional Neural Network (CNN)** on the MNIST dataset across multiple "Worker" nodes (GPUs), controlled by a central "Master" node.

### Key Features
* **Master-Worker Architecture:** A central scheduler dynamically assigns data batches to workers based on availability.
* **Straggler Mitigation:** The system detects slow nodes (simulated or real) and routes work to faster nodes to prevent bottlenecks.
* **Real-Time Monitoring:** Integrates with **NVIDIA NVML** to track GPU Temperature and Utilization during training.
* **Comparative Analysis:** Includes a built-in "Static Mode" to demonstrate the inefficiency of fixed workload assignment.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
* Python 3.8+
* Microsoft MPI (if on Windows) or MPICH (if on Linux/Mac)

### 2. Install Dependencies
Run the following command to install PyTorch, MPI4Py, and visualization tools:
```bash
pip install -r requirements.txt
3. Prepare the Dataset
Before running the cluster, download the MNIST dataset safely:

Bash

python download_data.py
🚀 How to Run
This project uses mpiexec to spawn multiple processes. We recommend running with 3 processes (1 Master + 2 Workers).

Mode A: Dynamic Scheduling (The "Smart" Way)
This runs the adaptive algorithm. Workers request tasks when they are free.

PowerShell

mpiexec -n 3 python -u src/main.py dynamic
Expected Result: Faster completion time. Fast workers process more batches; slow workers process fewer.

Mode B: Static Scheduling (The "Control" Group)
This runs the naive algorithm. Work is split 50/50 at the start.

PowerShell

mpiexec -n 3 python -u src/main.py static
Expected Result: Slower completion time. The fast worker finishes early and sits idle while the slow worker struggles.

📊 Visualizing Results
After running both modes, the system prints the total execution time in the terminal. You can generate performance comparison graphs using the included script.

Open src/visualize_results.py.

Update the real_static_time and real_dynamic_time variables with your specific run results.

Run the script:

Bash

python src/visualize_results.py
Check the plots/ folder for the generated charts.

📂 Project Structure
Plaintext

├── src/
│   ├── master/
│   │   └── scheduler.py    # Logic for Dynamic vs Static scheduling
│   ├── worker/
│   │   ├── trainer.py      # PyTorch training loop
│   │   └── monitor.py      # NVML GPU monitoring (Temp/Util)
│   ├── shared/
│   │   ├── model.py        # SimpleCNN architecture
│   │   └── data_loader.py  # MNIST dataset handling
│   └── main.py             # Entry point (initializes MPI)
├── download_data.py        # Helper script to setup data
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
👥 Contributors
Mubbara Majid - Architecture, Scheduling Logic, Visualization

Noor - Performance Monitoring Module (NVML Integration), Testing


---

### **Step 3: Push Everything to GitHub**
Now that your project is fully documented and working, do one final push:

```powershell
git add .
git commit -m "Final submission: Completed monitoring module and documentation"
git push