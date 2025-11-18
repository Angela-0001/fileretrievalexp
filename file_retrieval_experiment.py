"""
main.py - File Retrieval Performance Analysis
Main experiment file that uses the visualization module
"""

import time
import random
import statistics
import sys
import json
from datetime import datetime
import psutil
import os
import threading
from collections import defaultdict
import csv

# Import visualization module
from visualization import generate_all_visualizations, PerformanceVisualizer


class FileSystemSimulator:
    def __init__(self, num_files):
        self.num_files = num_files
        self.files_list = []
        self.files_index = {}
        self.binary_search_list = []
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.access_history = []
        self._generate_files()
    
    def _generate_files(self):
        print(f"⚙️  Generating {self.num_files} file records...")
        file_types = ['txt', 'pdf', 'doc', 'jpg', 'mp4', 'xlsx', 'pptx', 'zip']
        
        for i in range(self.num_files):
            file_id = f"FILE_{i:08d}"
            file_type = random.choice(file_types)
            
            file_data = {
                'id': file_id,
                'name': f"document_{i}.{file_type}",
                'size': random.randint(1024, 10485760),
                'path': f"/data/folder_{i % 100}/{file_id}",
                'created': datetime.now().isoformat(),
                'type': file_type,
                'access_count': 0,
                'last_access': None
            }
            
            self.files_list.append(file_data)
            self.files_index[file_id] = file_data
        
        self.binary_search_list = sorted(self.files_list, key=lambda x: x['id'])
        print(f"✓ Generated {self.num_files} files\n")
    
    def sequential_scan(self, file_id):
        """Linear search O(n)"""
        self.access_history.append(('sequential', file_id, time.time()))
        for file_data in self.files_list:
            if file_data['id'] == file_id:
                file_data['access_count'] += 1
                file_data['last_access'] = datetime.now().isoformat()
                return file_data
        return None
    
    def direct_access(self, file_id):
        """Hash table lookup O(1)"""
        self.access_history.append(('direct', file_id, time.time()))
        result = self.files_index.get(file_id, None)
        if result:
            result['access_count'] += 1
            result['last_access'] = datetime.now().isoformat()
        return result
    
    def binary_search(self, file_id):
        """Binary search O(log n)"""
        self.access_history.append(('binary', file_id, time.time()))
        left, right = 0, len(self.binary_search_list) - 1
        while left <= right:
            mid = (left + right) // 2
            if self.binary_search_list[mid]['id'] == file_id:
                result = self.binary_search_list[mid]
                result['access_count'] += 1
                result['last_access'] = datetime.now().isoformat()
                return result
            elif self.binary_search_list[mid]['id'] < file_id:
                left = mid + 1
            else:
                right = mid - 1
        return None
    
    def cached_access(self, file_id, cache_size=100):
        """Direct access with LRU cache"""
        self.access_history.append(('cached', file_id, time.time()))
        
        if file_id in self.cache:
            self.cache_hits += 1
            return self.cache[file_id]
        
        self.cache_misses += 1
        result = self.files_index.get(file_id, None)
        
        if result:
            if len(self.cache) >= cache_size:
                self.cache.pop(next(iter(self.cache)))
            self.cache[file_id] = result
            result['access_count'] += 1
        
        return result
    
    def get_cache_hit_rate(self):
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0
    
    def get_memory_usage(self):
        """Calculate memory usage of data structures"""
        list_size = sys.getsizeof(self.files_list)
        dict_size = sys.getsizeof(self.files_index)
        binary_size = sys.getsizeof(self.binary_search_list)
        cache_size = sys.getsizeof(self.cache)
        
        for item in self.files_list:
            list_size += sys.getsizeof(item)
        for key, val in self.files_index.items():
            dict_size += sys.getsizeof(key) + sys.getsizeof(val)
        for item in self.binary_search_list:
            binary_size += sys.getsizeof(item)
        for key, val in self.cache.items():
            cache_size += sys.getsizeof(key) + sys.getsizeof(val)
            
        return list_size, dict_size, binary_size, cache_size
    
    def analyze_file_types(self):
        """Analyze distribution of file types"""
        type_stats = defaultdict(lambda: {'count': 0, 'total_size': 0})
        
        for file_data in self.files_list:
            file_type = file_data['type']
            type_stats[file_type]['count'] += 1
            type_stats[file_type]['total_size'] += file_data['size']
        
        return dict(type_stats)
    
    def get_most_accessed_files(self, n=10):
        """Get top N most accessed files"""
        sorted_files = sorted(self.files_list, 
                            key=lambda x: x['access_count'], 
                            reverse=True)
        return sorted_files[:n]


def measure_performance_advanced(fs, file_ids, method_func, method_name):
    """Enhanced performance measurement"""
    times = []
    cpu_usage = []
    memory_usage = []
    
    process = psutil.Process(os.getpid())
    
    for file_id in file_ids:
        cpu_before = process.cpu_percent()
        mem_before = process.memory_info().rss / 1024 / 1024
        
        start_time = time.perf_counter()
        result = method_func(file_id)
        end_time = time.perf_counter()
        
        cpu_after = process.cpu_percent()
        mem_after = process.memory_info().rss / 1024 / 1024
        
        times.append((end_time - start_time) * 1000)
        cpu_usage.append((cpu_before + cpu_after) / 2)
        memory_usage.append(mem_after - mem_before)
    
    return {
        'avg': statistics.mean(times),
        'std': statistics.stdev(times) if len(times) > 1 else 0,
        'min': min(times),
        'max': max(times),
        'median': statistics.median(times),
        'cpu': statistics.mean(cpu_usage) if cpu_usage else 0,
        'memory_delta': statistics.mean(memory_usage),
        'all_times': times,
        'percentile_95': statistics.quantiles(times, n=20)[18] if len(times) > 20 else max(times)
    }


def concurrent_access_test(fs, file_ids, num_threads=4):
    """Test concurrent file access performance"""
    print(f"\n🔄 Testing concurrent access with {num_threads} threads...")
    
    results = []
    threads = []
    
    def worker(thread_id, ids):
        thread_times = []
        for file_id in ids:
            start = time.perf_counter()
            fs.direct_access(file_id)
            end = time.perf_counter()
            thread_times.append((end - start) * 1000)
        results.append({
            'thread_id': thread_id,
            'avg_time': statistics.mean(thread_times),
            'count': len(thread_times)
        })
    
    chunk_size = len(file_ids) // num_threads
    
    start_time = time.perf_counter()
    
    for i in range(num_threads):
        chunk = file_ids[i * chunk_size:(i + 1) * chunk_size]
        t = threading.Thread(target=worker, args=(i, chunk))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    total_time = (time.perf_counter() - start_time) * 1000
    
    return {
        'total_time': total_time,
        'threads': results,
        'avg_per_thread': statistics.mean([r['avg_time'] for r in results])
    }


def run_comprehensive_experiment(dataset_sizes, trials_per_size=100):
    """Run comprehensive experiment"""
    results = {
        'sizes': [],
        'sequential': [],
        'direct': [],
        'binary': [],
        'cached': [],
        'memory': [],
        'concurrent': [],
        'file_type_stats': []
    }
    
    print("="*80)
    print("🚀 ULTIMATE FILE RETRIEVAL PERFORMANCE EXPERIMENT")
    print("="*80)
    print(f"Methods: Sequential | Direct | Binary | Cached | Concurrent")
    print(f"Trials per dataset: {trials_per_size}\n")
    
    for size in dataset_sizes:
        print(f"\n{'='*80}")
        print(f"📊 Testing with {size:,} files")
        print(f"{'='*80}")
        
        fs = FileSystemSimulator(size)
        
        # Memory analysis
        list_mem, dict_mem, binary_mem, cache_mem = fs.get_memory_usage()
        memory_data = {
            'list': list_mem / 1024 / 1024,
            'dict': dict_mem / 1024 / 1024,
            'binary': binary_mem / 1024 / 1024,
            'cache': cache_mem / 1024 / 1024
        }
        
        # File type analysis
        file_type_stats = fs.analyze_file_types()
        
        search_ids = [f"FILE_{random.randint(0, size-1):08d}" 
                      for _ in range(trials_per_size)]
        
        # Test all methods
        print(f"\n⏱️  Testing Sequential Scan...")
        seq_results = measure_performance_advanced(fs, search_ids, 
                                                  fs.sequential_scan, "Sequential")
        
        print(f"⏱️  Testing Direct Access...")
        dir_results = measure_performance_advanced(fs, search_ids, 
                                                  fs.direct_access, "Direct")
        
        print(f"⏱️  Testing Binary Search...")
        bin_results = measure_performance_advanced(fs, search_ids, 
                                                  fs.binary_search, "Binary")
        
        print(f"⏱️  Testing Cached Access...")
        cache_results = measure_performance_advanced(fs, search_ids, 
                                                    fs.cached_access, "Cached")
        
        # Concurrent test
        concurrent_results = concurrent_access_test(fs, search_ids[:50], num_threads=4)
        
        results['sizes'].append(size)
        results['sequential'].append(seq_results)
        results['direct'].append(dir_results)
        results['binary'].append(bin_results)
        results['cached'].append(cache_results)
        results['memory'].append(memory_data)
        results['concurrent'].append(concurrent_results)
        results['file_type_stats'].append(file_type_stats)
        
        # Print summary
        print_summary(size, seq_results, dir_results, bin_results, 
                     cache_results, memory_data, fs, concurrent_results)
    
    return results, fs


def print_summary(size, seq, direct, binary, cached, memory, fs, concurrent):
    """Print detailed results summary"""
    print(f"\n{'─'*80}")
    print(f"📈 RESULTS FOR {size:,} FILES:")
    print(f"{'─'*80}")
    
    methods = [
        ('Sequential', seq, '🔴'),
        ('Direct', direct, '🟢'),
        ('Binary', binary, '🔵'),
        ('Cached', cached, '🟡')
    ]
    
    for name, res, icon in methods:
        print(f"\n{icon} {name}: {res['avg']:.4f} ms (±{res['std']:.4f})")
    
    print(f"\n💾 Memory: List={memory['list']:.2f}MB, Dict={memory['dict']:.2f}MB")
    print(f"📊 Cache Hit Rate: {fs.get_cache_hit_rate():.2f}%")
    
    speedup = seq['avg'] / direct['avg']
    print(f"🚀 Speedup (Direct): {speedup:.2f}x faster")
    print(f"{'─'*80}")


def export_to_csv(results, filename='performance_data.csv'):
    """Export results to CSV"""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset_Size', 'Method', 'Avg_Time_ms', 'Std_Dev', 
                        'Min_ms', 'Max_ms', 'Median_ms', 'CPU_%', '95th_Percentile'])
        
        methods = ['sequential', 'direct', 'binary', 'cached']
        for i, size in enumerate(results['sizes']):
            for method in methods:
                data = results[method][i]
                writer.writerow([
                    size, method, data['avg'], data['std'], 
                    data['min'], data['max'], data['median'], 
                    data['cpu'], data['percentile_95']
                ])
    
    print(f"✓ CSV exported to '{filename}'")


def export_to_json(results, filename='results.json'):
    """Export results to JSON"""
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'dataset_sizes': results['sizes'],
        'results': {}
    }
    
    for i, size in enumerate(results['sizes']):
        export_data['results'][str(size)] = {
            'sequential': {k: v for k, v in results['sequential'][i].items() if k != 'all_times'},
            'direct': {k: v for k, v in results['direct'][i].items() if k != 'all_times'},
            'binary': {k: v for k, v in results['binary'][i].items() if k != 'all_times'},
            'cached': {k: v for k, v in results['cached'][i].items() if k != 'all_times'},
            'memory_mb': results['memory'][i],
            'file_types': results['file_type_stats'][i]
        }
    
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"✓ JSON exported to '{filename}'")


def interactive_menu():
    """Interactive menu for running experiments"""
    print("\n" + "="*80)
    print("FILE RETRIEVAL PERFORMANCE ANALYZER")
    print("="*80)
    print("\n1. Quick Test (100, 1000, 5000 files)")
    print("2. Standard Test (100, 1000, 5000, 10000, 25000 files)")
    print("3. Extended Test (100, 1000, 5000, 10000, 25000, 50000 files)")
    print("4. Custom Test")
    print("5. Exit")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == '1':
        return [100, 1000, 5000], 100
    elif choice == '2':
        return [100, 1000, 5000, 10000, 25000], 100
    elif choice == '3':
        return [100, 1000, 5000, 10000, 25000, 50000], 100
    elif choice == '4':
        sizes_input = input("Enter dataset sizes (comma-separated): ")
        sizes = [int(x.strip()) for x in sizes_input.split(',')]
        trials = int(input("Enter number of trials per size (default 100): ") or "100")
        return sizes, trials
    else:
        return None, None


if __name__ == "__main__":
    print("\n🚀 FILE RETRIEVAL PERFORMANCE ANALYSIS SYSTEM")
    print("="*80)
    
    # Interactive menu or default configuration
    dataset_sizes, trials = interactive_menu()
    
    if dataset_sizes is None:
        print("\n👋 Exiting...")
        exit(0)
    
    # Run experiment
    print("\n🔬 Starting experiment...")
    results, fs = run_comprehensive_experiment(dataset_sizes, trials)
    
    # Export data
    print("\n📤 Exporting results...")
    export_to_csv(results)
    export_to_json(results)
    
    # Generate visualizations
    print("\n🎨 Generating visualizations...")
    generate_all_visualizations(results, fs)
    
    print("\n" + "="*80)
    print("✅ EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\n📁 Generated Files:")
    print("  • performance_data.csv - Raw data")
    print("  • results.json - Complete results")
    print("  • 00_executive_summary.png - Key findings")
    print("  • 01-09_*.png - Detailed analysis graphs")
    print("\n💡 Tip: Check all PNG files for comprehensive visual analysis!")
    print("="*80)