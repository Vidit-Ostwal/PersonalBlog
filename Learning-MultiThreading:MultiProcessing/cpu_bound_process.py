# Multiprocessing vs Multithreading: Interactive Guide
# A comprehensive educational notebook to understand when and how to use multiprocessing vs multithreading

import time
import threading
import multiprocessing as mp
import concurrent.futures
import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import psutil
import queue
from threading import Lock
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🚀 Multiprocessing vs Multithreading Interactive Guide")
print("=" * 60)

# ==============================================================================
# SECTION 2: VISUAL DEMONSTRATIONS
# ==============================================================================

print("\n\n🎬 SECTION 2: VISUAL DEMONSTRATIONS")
print("-" * 40)

# Utility functions for monitoring
def get_cpu_usage():
    """Get current CPU usage"""
    return psutil.cpu_percent(interval=0.1)

def monitor_performance(func, duration=5):
    """Monitor CPU usage during function execution"""
    cpu_usage = []
    start_time = time.time()
    
    def monitor():
        while time.time() - start_time < duration:
            cpu_usage.append(psutil.cpu_percent(interval=0.1))
            time.sleep(0.1)
    
    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    result = func()
    
    monitor_thread.join(timeout=1)
    return result, cpu_usage

# ==============================================================================
# DEMO A: CPU-BOUND TASK COMPARISON
# ==============================================================================

print("\n🔢 DEMO A: CPU-BOUND TASKS (Prime Number Calculation)")

def is_prime(n):
    """Check if a number is prime (CPU-intensive task)"""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def find_primes_sequential(start, end):
    """Find primes sequentially"""
    return [n for n in range(start, end) if is_prime(n)]

def find_primes_threading(start, end, num_threads=4):
    """Find primes using multithreading"""
    def worker(start_chunk, end_chunk, result_queue):
        primes = [n for n in range(start_chunk, end_chunk) if is_prime(n)]
        result_queue.put(primes)
    
    chunk_size = (end - start) // num_threads
    threads = []
    result_queue = queue.Queue()
    
    for i in range(num_threads):
        start_chunk = start + i * chunk_size
        end_chunk = start + (i + 1) * chunk_size if i < num_threads - 1 else end
        
        thread = threading.Thread(target=worker, args=(start_chunk, end_chunk, result_queue))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    all_primes = []
    while not result_queue.empty():
        all_primes.extend(result_queue.get())
    
    return sorted(all_primes)

# """Find primes using multiprocessing"""
def worker(start_chunk, end_chunk):
    return [n for n in range(start_chunk, end_chunk) if is_prime(n)]

def find_primes_multiprocessing(start, end, num_processes=4):
    chunk_size = (end - start) // num_processes
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        
        for i in range(num_processes):
            start_chunk = start + i * chunk_size
            end_chunk = start + (i + 1) * chunk_size if i < num_processes - 1 else end
            
            future = executor.submit(worker, start_chunk, end_chunk)
            futures.append(future)
        
        all_primes = []
        for future in concurrent.futures.as_completed(futures):
            all_primes.extend(future.result())
    
    return sorted(all_primes)

def compare_cpu_bound_performance():
    """Compare performance of different approaches for CPU-bound tasks"""
    print("\n⏱️  Performance Comparison (Finding primes from 1 to 5000000):")
    
    start_range, end_range = 1, 5000000
    results = {}
    
    # Sequential
    print("  🐌 Sequential processing...")
    start_time = time.time()
    seq_primes = find_primes_sequential(start_range, end_range)
    seq_time = time.time() - start_time
    results['Sequential'] = seq_time
    
    # Threading
    print("  🧵 Multithreading (4 threads)...")
    start_time = time.time()
    thread_primes = find_primes_threading(start_range, end_range, 4)
    thread_time = time.time() - start_time
    results['Threading'] = thread_time
    
    # Multiprocessing
    print("  🔄 Multiprocessing (4 processes)...")
    start_time = time.time()
    mp_primes = find_primes_multiprocessing(start_range, end_range, 4)
    mp_time = time.time() - start_time
    results['Multiprocessing'] = mp_time
    
    # Verify results are the same
    assert seq_primes == thread_primes == mp_primes, "Results don't match!"
    
    
    print(f"\n📊 Results ({len(seq_primes)} primes found):")
    for method, exec_time in results.items():
        speedup = seq_time / exec_time if method != 'Sequential' else 1.0
        print(f"  {method:15}: {exec_time:.3f}s (speedup: {speedup:.2f}x)")
    
    return results

def visualize_cpu_performance():
    """Create visualization of CPU-bound performance"""
    results = compare_cpu_bound_performance()
    
    plt.figure(figsize=(12, 5))
    
    # Performance comparison
    plt.subplot(1, 2, 1)
    methods = list(results.keys())
    times = list(results.values())
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4']
    
    bars = plt.bar(methods, times, color=colors, alpha=0.7)
    plt.title('CPU-Bound Task Performance\n(Prime Number Calculation)', fontsize=14, pad=20)
    plt.ylabel('Execution Time (seconds)')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{time_val:.3f}s', ha='center', va='bottom')
    
    # Speedup comparison
    plt.subplot(1, 2, 2)
    speedups = [times[0] / t for t in times]
    bars = plt.bar(methods, speedups, color=colors, alpha=0.7)
    plt.title('Speedup Factor\n(Higher is Better)', fontsize=14, pad=20)
    plt.ylabel('Speedup Factor')
    plt.xticks(rotation=45)
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Baseline')
    
    # Add value labels on bars
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{speedup:.2f}x', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print("\n💡 Key Insights:")
    print("  • Threading shows minimal improvement due to Python's GIL")
    print("  • Multiprocessing achieves true parallelism for CPU-bound tasks")
    print("  • The overhead of process creation is worth it for compute-intensive tasks")

# Run CPU-bound demonstration
if __name__ == "__main__":          # <---- guard here
    visualize_cpu_performance()