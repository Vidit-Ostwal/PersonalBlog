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
# DEMO B: I/O-BOUND TASK COMPARISON
# ==============================================================================

print("\n\n🌐 DEMO B: I/O-BOUND TASKS (Web Requests)")

def fetch_url(url):
    """Simulate fetching a URL (I/O-bound task)"""
    try:
        # Using httpbin for testing - it provides various endpoints for testing
        response = requests.get(f"https://httpbin.org/delay/{np.random.randint(1, 3)}", timeout=10)
        return f"✅ {response.status_code}"
    except:
        return "❌ Failed"

def fetch_urls_sequential(urls):
    """Fetch URLs sequentially"""
    results = []
    for url in urls:
        results.append(fetch_url(url))
    return results

def fetch_urls_threading(urls, num_threads=4):
    """Fetch URLs using multithreading"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(fetch_url, urls))
    return results

def fetch_urls_multiprocessing(urls, num_processes=4):
    """Fetch URLs using multiprocessing"""
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(fetch_url, urls))
    return results

def compare_io_bound_performance():
    """Compare performance of different approaches for I/O-bound tasks"""
    print("\n⏱️  Performance Comparison (10 HTTP requests with random delays):")
    
    # Create test URLs (using httpbin delay endpoint)
    urls = [f"https://httpbin.org/delay/1" for _ in range(10)]
    results = {}
    
    # Sequential
    print("  🐌 Sequential processing...")
    start_time = time.time()
    seq_results = fetch_urls_sequential(urls)
    seq_time = time.time() - start_time
    results['Sequential'] = seq_time
    
    # Threading
    print("  🧵 Multithreading (4 threads)...")
    start_time = time.time()
    thread_results = fetch_urls_threading(urls, 4)
    thread_time = time.time() - start_time
    results['Threading'] = thread_time
    
    # Multiprocessing
    print("  🔄 Multiprocessing (4 processes)...")
    start_time = time.time()
    mp_results = fetch_urls_multiprocessing(urls, 4)
    mp_time = time.time() - start_time
    results['Multiprocessing'] = mp_time
    
    print(f"\n📊 Results:")
    for method, exec_time in results.items():
        speedup = seq_time / exec_time if method != 'Sequential' else 1.0
        print(f"  {method:15}: {exec_time:.3f}s (speedup: {speedup:.2f}x)")
    
    return results

def simulate_io_task(num_task):
    """Simulate I/O with sleep"""
    time.sleep(0.1)  # Simulate 100ms I/O operation
    return "✅ Complete"

def simulate_io_comparison():
    """Simulate I/O comparison with local operations to avoid network dependency"""
    print("\n⏱️  Performance Comparison (Simulated I/O operations):")
    
    num_tasks = 20
    results = {}
    
    # Sequential
    print("  🐌 Sequential processing...")
    start_time = time.time()
    for _ in range(num_tasks):
        simulate_io_task(_)
    seq_time = time.time() - start_time
    results['Sequential'] = seq_time
    
    # Threading
    print("  🧵 Multithreading (4 threads)...")
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(simulate_io_task, range(num_tasks)))
    thread_time = time.time() - start_time
    results['Threading'] = thread_time
    
    # Multiprocessing
    print("  🔄 Multiprocessing (4 processes)...")
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        list(executor.map(simulate_io_task, range(num_tasks)))
    mp_time = time.time() - start_time
    results['Multiprocessing'] = mp_time
    
    return results

def visualize_io_performance():
    """Create visualization of I/O-bound performance"""
    results = simulate_io_comparison()
    print(results)
    
    plt.figure(figsize=(12, 5))
    
    # Performance comparison
    plt.subplot(1, 2, 1)
    methods = list(results.keys())
    times = list(results.values())
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4']
    
    bars = plt.bar(methods, times, color=colors, alpha=0.7)
    plt.title('I/O-Bound Task Performance\n(Simulated Network Requests)', fontsize=14, pad=20)
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
    print("  • Threading excels at I/O-bound tasks (no GIL limitation during I/O)")
    print("  • Multiprocessing has overhead that may not be worth it for simple I/O")
    print("  • Threading allows concurrent waiting for I/O operations")

# Run I/O-bound demonstration   
if __name__ == "__main__":          # <---- guard here
    visualize_io_performance()