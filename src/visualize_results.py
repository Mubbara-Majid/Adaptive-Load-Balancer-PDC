import matplotlib.pyplot as plt
import numpy as np
import os

def plot_comparison(static_time, dynamic_time, static_idle, dynamic_idle):
    labels = ['Static Scheduling', 'Adaptive (Dynamic) Scheduling']
    total_times = [static_time, dynamic_time]
    idle_times = [static_idle, dynamic_idle]

    x = np.arange(len(labels))  # label locations
    width = 0.35  # width of the bars

    # --- FIGURE 1: TOTAL TRAINING TIME ---
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x, total_times, width, label='Total Time', color=['#ff9999', '#66b3ff'])

    ax.set_ylabel('Time (Seconds)')
    ax.set_title('Total Training Duration: Static vs. Adaptive')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Function to auto-label the bars with value
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}s',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    
    # Save the plot
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig('plots/training_time_comparison.png')
    print("Saved plot to plots/training_time_comparison.png")
    plt.show()

    # --- FIGURE 2: GPU IDLE TIME (The "Straggler Effect") ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    rects2 = ax2.bar(x, idle_times, width, label='Avg GPU Idle Time', color=['#ffcc99', '#99ff99'])

    ax2.set_ylabel('Idle Time (Seconds)')
    ax2.set_title('Efficiency Analysis: Wasted GPU Time')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()

    autolabel(rects2)
    plt.savefig('plots/idle_time_comparison.png')
    print("Saved plot to plots/idle_time_comparison.png")
    plt.show()

if __name__ == "__main__":    
    real_static_time = 47.24
    real_dynamic_time = 25.87
    
    real_static_idle_avg = 20.0 
    real_dynamic_idle_avg = 0.1
    
    plot_comparison(real_static_time, real_dynamic_time, real_static_idle_avg, real_dynamic_idle_avg)