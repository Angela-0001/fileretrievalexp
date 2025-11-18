"""
visualization.py - Advanced Visualization Module
Separate file for all graphing and visual analysis methods
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
try:
    import seaborn as sns  # type: ignore[reportMissingModuleSource]
    _HAS_SEABORN = True
except Exception:
    sns = None
    _HAS_SEABORN = False
    # seaborn not installed; we'll fall back to matplotlib styles
    try:
        plt.style.use('seaborn-whitegrid')
    except Exception:
        plt.style.use('default')
from mpl_toolkits.mplot3d import Axes3D

# Set style
if _HAS_SEABORN:
    try:
        sns.set_style("whitegrid")
    except Exception:
        pass
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'


class PerformanceVisualizer:
    """Class containing all visualization methods"""
    
    def __init__(self, results, fs=None):
        self.results = results
        self.fs = fs
        self.colors = {
            'sequential': '#e74c3c',
            'direct': '#27ae60',
            'binary': '#3498db',
            'cached': '#f39c12'
        }
    
    def create_all_visualizations(self):
        """Generate all visualizations in one call"""
        print("\n🎨 Generating comprehensive visualizations...")
        
        self.plot_performance_comparison()
        self.plot_speedup_analysis()
        self.plot_memory_analysis()
        self.plot_distribution_analysis()
        self.plot_efficiency_heatmap()
        self.plot_3d_performance_surface()
        self.plot_time_series_analysis()
        self.plot_comparative_dashboard()
        self.plot_advanced_metrics()
        
        print("✅ All visualizations generated successfully!")
    
    def plot_performance_comparison(self):
        """Main performance comparison chart"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        sizes = self.results['sizes']
        
        # Linear Scale
        for method in ['sequential', 'direct', 'binary', 'cached']:
            avg_times = [r['avg'] for r in self.results[method]]
            ax1.plot(sizes, avg_times, 'o-', linewidth=2.5, markersize=9,
                    label=method.capitalize(), color=self.colors[method])
        
        ax1.set_xlabel('Number of Files', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Average Retrieval Time (ms)', fontsize=13, fontweight='bold')
        ax1.set_title('Performance Comparison - Linear Scale', fontsize=15, fontweight='bold')
        ax1.legend(fontsize=11, loc='upper left')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Log Scale
        for method in ['sequential', 'direct', 'binary', 'cached']:
            avg_times = [r['avg'] for r in self.results[method]]
            ax2.plot(sizes, avg_times, 'o-', linewidth=2.5, markersize=9,
                    label=method.capitalize(), color=self.colors[method])
        
        ax2.set_xlabel('Number of Files', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Average Retrieval Time (ms)', fontsize=13, fontweight='bold')
        ax2.set_title('Performance Comparison - Log Scale', fontsize=15, fontweight='bold')
        ax2.legend(fontsize=11, loc='upper left')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('01_performance_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 01_performance_comparison.png")
        plt.close()
    
    def plot_speedup_analysis(self):
        """Detailed speedup analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        sizes = self.results['sizes']
        seq_avg = [r['avg'] for r in self.results['sequential']]
        
        # 1. Speedup Bar Chart
        x = np.arange(len(sizes))
        width = 0.25
        
        for i, method in enumerate(['direct', 'binary', 'cached']):
            method_avg = [r['avg'] for r in self.results[method]]
            speedup = [seq_avg[j]/method_avg[j] for j in range(len(sizes))]
            offset = (i - 1) * width
            ax1.bar(x + offset, speedup, width, label=method.capitalize(),
                   color=self.colors[method], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax1.set_xlabel('Dataset Size', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Speedup Factor (x)', fontsize=12, fontweight='bold')
        ax1.set_title('Speedup vs Sequential Scan', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{s:,}" for s in sizes], rotation=45, ha='right')
        ax1.legend(fontsize=11)
        ax1.grid(True, axis='y', alpha=0.3)
        
        # 2. Speedup Line Chart
        for method in ['direct', 'binary', 'cached']:
            method_avg = [r['avg'] for r in self.results[method]]
            speedup = [seq_avg[i]/method_avg[i] for i in range(len(sizes))]
            ax2.plot(sizes, speedup, 'o-', linewidth=2.5, markersize=8,
                    label=method.capitalize(), color=self.colors[method])
        
        ax2.set_xlabel('Number of Files', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Speedup Factor (x)', fontsize=12, fontweight='bold')
        ax2.set_title('Speedup Trend Analysis', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        # 3. Relative Performance (Normalized)
        for method in ['sequential', 'direct', 'binary', 'cached']:
            avg_times = [r['avg'] for r in self.results[method]]
            normalized = [(t / avg_times[0]) * 100 for t in avg_times]
            ax3.plot(sizes, normalized, 'o-', linewidth=2, markersize=8,
                    label=method.capitalize(), color=self.colors[method])
        
        ax3.set_xlabel('Number of Files', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Relative Time (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Normalized Performance (Base: Smallest Dataset)', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # 4. Efficiency Score
        for method in ['direct', 'binary', 'cached']:
            method_avg = [r['avg'] for r in self.results[method]]
            efficiency = [(seq_avg[i] - method_avg[i]) / seq_avg[i] * 100 
                         for i in range(len(sizes))]
            ax4.plot(sizes, efficiency, 'o-', linewidth=2.5, markersize=8,
                    label=method.capitalize(), color=self.colors[method])
        
        ax4.set_xlabel('Number of Files', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Efficiency Improvement (%)', fontsize=12, fontweight='bold')
        ax4.set_title('Time Saved vs Sequential', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='red', linestyle='--', linewidth=1)
        
        plt.tight_layout()
        plt.savefig('02_speedup_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 02_speedup_analysis.png")
        plt.close()
    
    def plot_memory_analysis(self):
        """Memory usage visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        sizes = self.results['sizes']
        
        # 1. Memory Usage by Method
        for method in ['list', 'dict', 'binary', 'cache']:
            memory = [m[method] for m in self.results['memory']]
            color = self.colors.get(method, '#95a5a6')
            ax1.plot(sizes, memory, 'o-', linewidth=2.5, markersize=8,
                    label=method.capitalize(), color=color)
        
        ax1.set_xlabel('Number of Files', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Memory Usage (MB)', fontsize=12, fontweight='bold')
        ax1.set_title('Memory Consumption by Method', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 2. Memory Efficiency (MB per 1000 files)
        for method in ['list', 'dict', 'binary', 'cache']:
            memory = [m[method] for m in self.results['memory']]
            efficiency = [(memory[i] / (sizes[i] / 1000)) for i in range(len(sizes))]
            color = self.colors.get(method, '#95a5a6')
            ax2.plot(sizes, efficiency, 'o-', linewidth=2, markersize=8,
                    label=method.capitalize(), color=color)
        
        ax2.set_xlabel('Number of Files', fontsize=12, fontweight='bold')
        ax2.set_ylabel('MB per 1000 Files', fontsize=12, fontweight='bold')
        ax2.set_title('Memory Efficiency', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # 3. Stacked Area Chart
        list_mem = [m['list'] for m in self.results['memory']]
        dict_mem = [m['dict'] for m in self.results['memory']]
        binary_mem = [m['binary'] for m in self.results['memory']]
        cache_mem = [m['cache'] for m in self.results['memory']]
        
        ax3.fill_between(sizes, 0, list_mem, alpha=0.7, color=self.colors['sequential'], label='List')
        ax3.fill_between(sizes, list_mem, 
                        [list_mem[i]+dict_mem[i] for i in range(len(sizes))],
                        alpha=0.7, color=self.colors['direct'], label='Dict')
        ax3.fill_between(sizes, 
                        [list_mem[i]+dict_mem[i] for i in range(len(sizes))],
                        [list_mem[i]+dict_mem[i]+binary_mem[i] for i in range(len(sizes))],
                        alpha=0.7, color=self.colors['binary'], label='Binary')
        
        ax3.set_xlabel('Number of Files', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Cumulative Memory (MB)', fontsize=12, fontweight='bold')
        ax3.set_title('Cumulative Memory Usage', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # 4. Memory vs Speed Trade-off
        for i, size in enumerate(sizes):
            methods_data = []
            for method_name, mem_key in [('Sequential', 'list'), ('Direct', 'dict'), 
                                         ('Binary', 'binary'), ('Cached', 'cache')]:
                method_key = method_name.lower()
                if method_key in self.results:
                    time = self.results[method_key][i]['avg']
                    memory = self.results['memory'][i][mem_key]
                    methods_data.append((memory, time, method_name))
            
            for memory, time, label in methods_data:
                color = self.colors.get(label.lower(), '#95a5a6')
                ax4.scatter(memory, time, s=200, alpha=0.6, color=color, 
                          edgecolors='black', linewidth=1.5)
                ax4.annotate(f"{label}\n({size})", (memory, time), 
                           fontsize=8, ha='center')
        
        ax4.set_xlabel('Memory Usage (MB)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Average Time (ms)', fontsize=12, fontweight='bold')
        ax4.set_title('Memory-Speed Trade-off Analysis', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('03_memory_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 03_memory_analysis.png")
        plt.close()
    
    def plot_distribution_analysis(self):
        """Statistical distribution analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Box plots for each dataset size
        for idx, size in enumerate(self.results['sizes'][-4:]):  # Last 4 sizes
            ax = axes[idx // 2, idx % 2]
            
            data_to_plot = []
            labels = []
            colors_list = []
            
            for method in ['sequential', 'direct', 'binary', 'cached']:
                if method in self.results:
                    size_idx = self.results['sizes'].index(size)
                    times = self.results[method][size_idx]['all_times']
                    data_to_plot.append(times)
                    labels.append(method.capitalize())
                    colors_list.append(self.colors[method])
            
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                           showmeans=True, meanline=True)
            
            for patch, color in zip(bp['boxes'], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel('Time (ms)', fontsize=11, fontweight='bold')
            ax.set_title(f'Distribution - {size:,} Files', fontsize=13, fontweight='bold')
            ax.grid(True, axis='y', alpha=0.3)
            ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('04_distribution_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 04_distribution_analysis.png")
        plt.close()
    
    def plot_efficiency_heatmap(self):
        """Create efficiency heatmap"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        sizes = self.results['sizes']
        methods = ['sequential', 'direct', 'binary', 'cached']
        
        # 1. Time Heatmap
        time_matrix = []
        for method in methods:
            avg_times = [r['avg'] for r in self.results[method]]
            time_matrix.append(avg_times)
        
        im1 = ax1.imshow(time_matrix, cmap='RdYlGn_r', aspect='auto')
        ax1.set_xticks(range(len(sizes)))
        ax1.set_xticklabels([f"{s:,}" for s in sizes], rotation=45, ha='right')
        ax1.set_yticks(range(len(methods)))
        ax1.set_yticklabels([m.capitalize() for m in methods])
        ax1.set_xlabel('Dataset Size', fontsize=12, fontweight='bold')
        ax1.set_title('Performance Heatmap (ms)', fontsize=14, fontweight='bold')
        
        # Add values
        for i in range(len(methods)):
            for j in range(len(sizes)):
                text = ax1.text(j, i, f'{time_matrix[i][j]:.2f}',
                              ha="center", va="center", color="black", fontsize=9)
        
        plt.colorbar(im1, ax=ax1, label='Time (ms)')
        
        # 2. Speedup Heatmap
        seq_times = time_matrix[0]
        speedup_matrix = []
        for method_times in time_matrix:
            speedup_row = [seq_times[i] / method_times[i] for i in range(len(sizes))]
            speedup_matrix.append(speedup_row)
        
        im2 = ax2.imshow(speedup_matrix, cmap='RdYlGn', aspect='auto')
        ax2.set_xticks(range(len(sizes)))
        ax2.set_xticklabels([f"{s:,}" for s in sizes], rotation=45, ha='right')
        ax2.set_yticks(range(len(methods)))
        ax2.set_yticklabels([m.capitalize() for m in methods])
        ax2.set_xlabel('Dataset Size', fontsize=12, fontweight='bold')
        ax2.set_title('Speedup Heatmap (x)', fontsize=14, fontweight='bold')
        
        # Add values
        for i in range(len(methods)):
            for j in range(len(sizes)):
                text = ax2.text(j, i, f'{speedup_matrix[i][j]:.1f}x',
                              ha="center", va="center", color="black", fontsize=9)
        
        plt.colorbar(im2, ax=ax2, label='Speedup (x)')
        
        plt.tight_layout()
        plt.savefig('05_efficiency_heatmap.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 05_efficiency_heatmap.png")
        plt.close()
    
    def plot_3d_performance_surface(self):
        """3D surface plot of performance"""
        fig = plt.figure(figsize=(16, 10))
        
        sizes = self.results['sizes']
        methods = ['sequential', 'direct', 'binary', 'cached']
        
        for idx, method in enumerate(methods):
            ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
            
            # Create mesh
            X = sizes
            Y = range(len(self.results[method][0]['all_times'][:50]))  # First 50 trials
            X, Y = np.meshgrid(X, Y)
            
            Z = []
            for trial in range(len(Y)):
                trial_times = []
                for size_idx in range(len(sizes)):
                    if trial < len(self.results[method][size_idx]['all_times']):
                        trial_times.append(self.results[method][size_idx]['all_times'][trial])
                    else:
                        trial_times.append(self.results[method][size_idx]['avg'])
                Z.append(trial_times)
            
            Z = np.array(Z)
            
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
            
            ax.set_xlabel('Dataset Size', fontsize=10, fontweight='bold')
            ax.set_ylabel('Trial Number', fontsize=10, fontweight='bold')
            ax.set_zlabel('Time (ms)', fontsize=10, fontweight='bold')
            ax.set_title(f'{method.capitalize()} - 3D Performance', fontsize=12, fontweight='bold')
            
            fig.colorbar(surf, ax=ax, shrink=0.5)
        
        plt.tight_layout()
        plt.savefig('06_3d_performance_surface.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 06_3d_performance_surface.png")
        plt.close()
    
    def plot_time_series_analysis(self):
        """Time series analysis of access patterns"""
        if not self.fs or not hasattr(self.fs, 'access_history'):
            print("⚠ Skipping time series - no access history available")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Parse access history
        history = self.fs.access_history[-1000:]  # Last 1000 accesses
        
        if not history:
            print("⚠ No access history data available")
            return
        
        methods_count = {'sequential': 0, 'direct': 0, 'binary': 0, 'cached': 0}
        for method, _, _ in history:
            if method in methods_count:
                methods_count[method] += 1
        
        # 1. Access Method Distribution
        ax1.pie(methods_count.values(), labels=methods_count.keys(), autopct='%1.1f%%',
               colors=[self.colors[m] for m in methods_count.keys()], startangle=90)
        ax1.set_title('Access Method Distribution', fontsize=14, fontweight='bold')
        
        # 2. Access Timeline
        method_colors = {'sequential': self.colors['sequential'], 
                        'direct': self.colors['direct'],
                        'binary': self.colors['binary'], 
                        'cached': self.colors['cached']}
        
        for i, (method, file_id, timestamp) in enumerate(history[:200]):
            color = method_colors.get(method, 'gray')
            ax2.scatter(i, 0, c=color, s=50, alpha=0.6)
        
        ax2.set_xlabel('Access Sequence', fontsize=12, fontweight='bold')
        ax2.set_title('Access Pattern Timeline (First 200)', fontsize=14, fontweight='bold')
        ax2.set_yticks([])
        ax2.grid(True, alpha=0.3)
        
        # Create legend
        legend_elements = [mpatches.Patch(color=color, label=method.capitalize())
                          for method, color in method_colors.items()]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        # 3. Most Accessed Files
        if hasattr(self.fs, 'get_most_accessed_files'):
            top_files = self.fs.get_most_accessed_files(15)
            file_names = [f['name'][:20] for f in top_files]
            access_counts = [f['access_count'] for f in top_files]
            
            bars = ax3.barh(range(len(file_names)), access_counts, 
                           color='#9b59b6', alpha=0.8, edgecolor='black')
            ax3.set_yticks(range(len(file_names)))
            ax3.set_yticklabels(file_names, fontsize=9)
            ax3.set_xlabel('Access Count', fontsize=12, fontweight='bold')
            ax3.set_title('Most Accessed Files (Top 15)', fontsize=14, fontweight='bold')
            ax3.grid(True, axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, count) in enumerate(zip(bars, access_counts)):
                ax3.text(count, i, f' {count}', va='center', fontsize=9)
        
        # 4. File Type Distribution
        if hasattr(self.fs, 'analyze_file_types'):
            type_stats = self.fs.analyze_file_types()
            types = list(type_stats.keys())
            counts = [type_stats[t]['count'] for t in types]
            sizes_mb = [type_stats[t]['total_size'] / 1024 / 1024 for t in types]
            
            x = range(len(types))
            width = 0.35
            
            ax4_twin = ax4.twinx()
            
            bars1 = ax4.bar([i - width/2 for i in x], counts, width, 
                           label='File Count', color='#3498db', alpha=0.8)
            bars2 = ax4_twin.bar([i + width/2 for i in x], sizes_mb, width,
                                label='Total Size (MB)', color='#e74c3c', alpha=0.8)
            
            ax4.set_xlabel('File Type', fontsize=12, fontweight='bold')
            ax4.set_ylabel('File Count', fontsize=11, fontweight='bold', color='#3498db')
            ax4_twin.set_ylabel('Total Size (MB)', fontsize=11, fontweight='bold', color='#e74c3c')
            ax4.set_title('File Type Statistics', fontsize=14, fontweight='bold')
            ax4.set_xticks(x)
            ax4.set_xticklabels(types, rotation=45, ha='right')
            ax4.grid(True, alpha=0.3)
            
            # Legends
            ax4.legend(loc='upper left')
            ax4_twin.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig('07_time_series_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 07_time_series_analysis.png")
        plt.close()
    
    def plot_comparative_dashboard(self):
        """Create comprehensive dashboard"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        sizes = self.results['sizes']
        
        # 1. Main Performance Chart (Large, top-left)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        for method in ['sequential', 'direct', 'binary', 'cached']:
            avg_times = [r['avg'] for r in self.results[method]]
            ax1.plot(sizes, avg_times, 'o-', linewidth=3, markersize=10,
                    label=method.capitalize(), color=self.colors[method])
        
        ax1.set_xlabel('Number of Files', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Average Time (ms)', fontsize=14, fontweight='bold')
        ax1.set_title('PRIMARY PERFORMANCE COMPARISON', fontsize=16, fontweight='bold')
        ax1.legend(fontsize=12, loc='upper left')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # 2. Speedup Summary (top-right)
        ax2 = fig.add_subplot(gs[0, 2])
        seq_avg = [r['avg'] for r in self.results['sequential']]
        final_speedups = []
        for method in ['direct', 'binary', 'cached']:
            method_avg = [r['avg'] for r in self.results[method]]
            speedup = seq_avg[-1] / method_avg[-1]
            final_speedups.append(speedup)
        
        bars = ax2.bar(range(3), final_speedups, 
                      color=[self.colors['direct'], self.colors['binary'], self.colors['cached']],
                      alpha=0.8, edgecolor='black', linewidth=2)
        ax2.set_xticks(range(3))
        ax2.set_xticklabels(['Direct', 'Binary', 'Cached'], fontsize=10)
        ax2.set_ylabel('Speedup (x)', fontsize=11, fontweight='bold')
        ax2.set_title(f'Final Speedup\n({sizes[-1]:,} files)', fontsize=12, fontweight='bold')
        ax2.grid(True, axis='y', alpha=0.3)
        
        for i, (bar, val) in enumerate(zip(bars, final_speedups)):
            ax2.text(i, val, f'{val:.1f}x', ha='center', va='bottom', 
                    fontweight='bold', fontsize=11)
        
        # 3. Memory Usage (middle-right)
        ax3 = fig.add_subplot(gs[1, 2])
        final_memory = self.results['memory'][-1]
        mem_values = [final_memory['list'], final_memory['dict'], 
                     final_memory['binary'], final_memory['cache']]
        mem_labels = ['Sequential', 'Direct', 'Binary', 'Cached']
        
        wedges, texts, autotexts = ax3.pie(mem_values, labels=mem_labels, autopct='%1.1f%%',
               colors=[self.colors[m.lower()] for m in mem_labels], startangle=90)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax3.set_title(f'Memory Distribution\n({sizes[-1]:,} files)', fontsize=12, fontweight='bold')
        
        # 4. Efficiency Trend (bottom-left)
        ax4 = fig.add_subplot(gs[2, 0])
        for method in ['direct', 'binary', 'cached']:
            method_avg = [r['avg'] for r in self.results[method]]
            efficiency = [(seq_avg[i] - method_avg[i]) / seq_avg[i] * 100 
                         for i in range(len(sizes))]
            ax4.plot(sizes, efficiency, 'o-', linewidth=2, markersize=7,
                    label=method.capitalize(), color=self.colors[method])
        
        ax4.set_xlabel('Files', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Efficiency (%)', fontsize=11, fontweight='bold')
        ax4.set_title('Efficiency Improvement', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # 5. CPU Usage (bottom-middle)
        ax5 = fig.add_subplot(gs[2, 1])
        x = range(len(sizes))
        width = 0.2
        
        for i, method in enumerate(['sequential', 'direct', 'binary', 'cached']):
            cpu_values = [r['cpu'] for r in self.results[method]]
            offset = (i - 1.5) * width
            ax5.bar([j + offset for j in x], cpu_values, width,
                   label=method.capitalize(), color=self.colors[method], alpha=0.8)
        
        ax5.set_xlabel('Dataset', fontsize=11, fontweight='bold')
        ax5.set_ylabel('CPU %', fontsize=11, fontweight='bold')
        ax5.set_title('CPU Utilization', fontsize=12, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels([f"{s//1000}K" for s in sizes], fontsize=9)
        ax5.legend(fontsize=9)
        ax5.grid(True, axis='y', alpha=0.3)
        
        # 6. Statistical Summary (bottom-right)
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        summary_text = "PERFORMANCE SUMMARY\n" + "="*30 + "\n\n"
        summary_text += f"Datasets Tested: {len(sizes)}\n"
        summary_text += f"Range: {sizes[0]:,} - {sizes[-1]:,} files\n"
        summary_text += f"Trials per size: {len(self.results['sequential'][0]['all_times'])}\n\n"
        
        summary_text += "BEST METHOD:\n"
        best_method = min(['direct', 'binary', 'cached'], 
                         key=lambda m: self.results[m][-1]['avg'])
        summary_text += f"  {best_method.capitalize()}\n"
        summary_text += f"  {self.results[best_method][-1]['avg']:.4f} ms\n\n"
        
        summary_text += "MAX SPEEDUP:\n"
        max_speedup = max(final_speedups)
        summary_text += f"  {max_speedup:.1f}x faster\n\n"
        
        if self.fs and hasattr(self.fs, 'get_cache_hit_rate'):
            summary_text += f"CACHE HIT RATE:\n"
            summary_text += f"  {self.fs.get_cache_hit_rate():.1f}%\n\n"
        
        summary_text += "RECOMMENDATION:\n"
        summary_text += f"  Use {best_method.capitalize()}\n"
        summary_text += f"  for optimal performance"
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('COMPREHENSIVE PERFORMANCE DASHBOARD', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.savefig('08_comparative_dashboard.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 08_comparative_dashboard.png")
        plt.close()
    
    def plot_advanced_metrics(self):
        """Advanced statistical metrics visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        sizes = self.results['sizes']
        
        # 1. Variance Analysis
        for method in ['sequential', 'direct', 'binary', 'cached']:
            std_devs = [r['std'] for r in self.results[method]]
            ax1.plot(sizes, std_devs, 'o-', linewidth=2.5, markersize=8,
                    label=method.capitalize(), color=self.colors[method])
        
        ax1.set_xlabel('Number of Files', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Standard Deviation (ms)', fontsize=12, fontweight='bold')
        ax1.set_title('Performance Variance Analysis', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 2. Min/Max Range
        for method in ['sequential', 'direct', 'binary', 'cached']:
            mins = [r['min'] for r in self.results[method]]
            maxs = [r['max'] for r in self.results[method]]
            avgs = [r['avg'] for r in self.results[method]]
            
            ax2.fill_between(sizes, mins, maxs, alpha=0.3, color=self.colors[method])
            ax2.plot(sizes, avgs, 'o-', linewidth=2, markersize=7,
                    label=method.capitalize(), color=self.colors[method])
        
        ax2.set_xlabel('Number of Files', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
        ax2.set_title('Min/Max Range with Average', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # 3. 95th Percentile Latency
        for method in ['sequential', 'direct', 'binary', 'cached']:
            p95 = [r['percentile_95'] for r in self.results[method]]
            avgs = [r['avg'] for r in self.results[method]]
            
            ax3.plot(sizes, p95, 's-', linewidth=2.5, markersize=8,
                    label=f'{method.capitalize()} (95th)', 
                    color=self.colors[method], alpha=0.7)
            ax3.plot(sizes, avgs, 'o--', linewidth=1.5, markersize=6,
                    label=f'{method.capitalize()} (avg)', 
                    color=self.colors[method], alpha=0.4)
        
        ax3.set_xlabel('Number of Files', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
        ax3.set_title('95th Percentile vs Average Latency', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=9, ncol=2)
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # 4. Coefficient of Variation (CV)
        for method in ['sequential', 'direct', 'binary', 'cached']:
            cvs = []
            for r in self.results[method]:
                cv = (r['std'] / r['avg']) * 100 if r['avg'] > 0 else 0
                cvs.append(cv)
            
            ax4.plot(sizes, cvs, 'o-', linewidth=2.5, markersize=8,
                    label=method.capitalize(), color=self.colors[method])
        
        ax4.set_xlabel('Number of Files', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
        ax4.set_title('Performance Consistency (Lower is Better)', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=10, color='red', linestyle='--', linewidth=1, alpha=0.5, label='10% threshold')
        
        plt.tight_layout()
        plt.savefig('09_advanced_metrics.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 09_advanced_metrics.png")
        plt.close()
    
    def create_animated_visualization(self):
        """Create frame-by-frame comparison (saved as individual frames)"""
        print("\n🎬 Creating animation frames...")
        
        sizes = self.results['sizes']
        
        for frame_idx, size in enumerate(sizes):
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot all methods up to current size
            for method in ['sequential', 'direct', 'binary', 'cached']:
                times = [self.results[method][i]['avg'] for i in range(frame_idx + 1)]
                current_sizes = sizes[:frame_idx + 1]
                
                ax.plot(current_sizes, times, 'o-', linewidth=3, markersize=10,
                       label=method.capitalize(), color=self.colors[method])
            
            ax.set_xlabel('Number of Files', fontsize=14, fontweight='bold')
            ax.set_ylabel('Average Time (ms)', fontsize=14, fontweight='bold')
            ax.set_title(f'Performance Evolution - Up to {size:,} Files', 
                        fontsize=16, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_yscale('log')
            
            # Set consistent axis limits
            ax.set_xlim([sizes[0] * 0.8, sizes[-1] * 1.2])
            all_times = []
            for method in ['sequential', 'direct', 'binary', 'cached']:
                all_times.extend([r['avg'] for r in self.results[method]])
            ax.set_ylim([min(all_times) * 0.5, max(all_times) * 1.5])
            
            plt.tight_layout()
            plt.savefig(f'animation_frame_{frame_idx:02d}.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"✓ Created {len(sizes)} animation frames")
        print("  Tip: Use a tool like ffmpeg to combine frames into a video")
        print("  Command: ffmpeg -framerate 1 -i animation_frame_%02d.png -c:v libx264 animation.mp4")


def create_summary_report_image(results, fs=None):
    """Create a single summary image with key metrics"""
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('white')
    
    # Title
    fig.text(0.5, 0.95, 'FILE RETRIEVAL PERFORMANCE - EXECUTIVE SUMMARY', 
            ha='center', fontsize=20, fontweight='bold')
    
    sizes = results['sizes']
    seq_avg = [r['avg'] for r in results['sequential']]
    dir_avg = [r['avg'] for r in results['direct']]
    
    # Left: Main comparison
    ax1 = plt.subplot(2, 2, 1)
    for method in ['sequential', 'direct', 'binary', 'cached']:
        avg_times = [r['avg'] for r in results[method]]
        ax1.plot(sizes, avg_times, 'o-', linewidth=3, markersize=9,
                label=method.capitalize(), 
                color={'sequential': '#e74c3c', 'direct': '#27ae60', 
                      'binary': '#3498db', 'cached': '#f39c12'}[method])
    
    ax1.set_xlabel('Number of Files', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Performance Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Right: Speedup
    ax2 = plt.subplot(2, 2, 2)
    speedup_direct = [seq_avg[i]/dir_avg[i] for i in range(len(sizes))]
    ax2.plot(sizes, speedup_direct, 'o-', linewidth=3, markersize=9, 
            color='#27ae60', label='Direct Access')
    ax2.fill_between(sizes, 1, speedup_direct, alpha=0.3, color='#27ae60')
    ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Baseline')
    
    ax2.set_xlabel('Number of Files', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speedup Factor (x)', fontsize=12, fontweight='bold')
    ax2.set_title('Direct Access Speedup', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # Bottom left: Key metrics
    ax3 = plt.subplot(2, 2, 3)
    ax3.axis('off')
    
    metrics_text = "KEY PERFORMANCE INDICATORS\n" + "="*40 + "\n\n"
    metrics_text += f"Dataset Range: {sizes[0]:,} - {sizes[-1]:,} files\n\n"
    
    final_speedup = seq_avg[-1] / dir_avg[-1]
    metrics_text += f"Maximum Speedup: {final_speedup:.1f}x\n"
    metrics_text += f"  (Direct Access vs Sequential)\n\n"
    
    metrics_text += f"Sequential Time (largest): {seq_avg[-1]:.2f} ms\n"
    metrics_text += f"Direct Access Time: {dir_avg[-1]:.4f} ms\n\n"
    
    if fs and hasattr(fs, 'get_cache_hit_rate'):
        metrics_text += f"Cache Hit Rate: {fs.get_cache_hit_rate():.1f}%\n\n"
    
    metrics_text += "Time Complexity:\n"
    metrics_text += "  • Sequential: O(n)\n"
    metrics_text += "  • Direct: O(1)\n"
    metrics_text += "  • Binary: O(log n)\n\n"
    
    metrics_text += "RECOMMENDATION:\n"
    metrics_text += "Use Direct Access (Hash Index)\n"
    metrics_text += "for best performance!"
    
    ax3.text(0.1, 0.9, metrics_text, transform=ax3.transAxes,
            fontsize=12, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='#e8f8f5', alpha=0.8, pad=1))
    
    # Bottom right: Memory
    ax4 = plt.subplot(2, 2, 4)
    final_memory = results['memory'][-1]
    labels = ['Sequential\nList', 'Direct\nAccess', 'Binary\nSearch', 'Cache']
    values = [final_memory['list'], final_memory['dict'], 
              final_memory['binary'], final_memory['cache']]
    colors = ['#e74c3c', '#27ae60', '#3498db', '#f39c12']
    
    bars = ax4.bar(range(4), values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax4.set_ylabel('Memory Usage (MB)', fontsize=12, fontweight='bold')
    ax4.set_title('Memory Consumption', fontsize=14, fontweight='bold')
    ax4.set_xticks(range(4))
    ax4.set_xticklabels(labels, fontsize=10)
    ax4.grid(True, axis='y', alpha=0.3)
    
    for bar, val in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2, val, f'{val:.1f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('00_executive_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 00_executive_summary.png")
    plt.close()


# Convenience function to generate all visualizations
def generate_all_visualizations(results, fs=None):
    """Generate all visualizations at once"""
    print("\n" + "="*70)
    print("🎨 GENERATING ALL VISUALIZATIONS")
    print("="*70)
    
    # Create summary first
    create_summary_report_image(results, fs)
    
    # Create visualizer and generate all plots
    visualizer = PerformanceVisualizer(results, fs)
    visualizer.create_all_visualizations()
    
    print("\n" + "="*70)
    print("✅ ALL VISUALIZATIONS COMPLETED!")
    print("="*70)
    print("\nGenerated files:")
    print("  00_executive_summary.png")
    print("  01_performance_comparison.png")
    print("  02_speedup_analysis.png")
    print("  03_memory_analysis.png")
    print("  04_distribution_analysis.png")
    print("  05_efficiency_heatmap.png")
    print("  06_3d_performance_surface.png")
    print("  07_time_series_analysis.png")
    print("  08_comparative_dashboard.png")
    print("  09_advanced_metrics.png")
    print("  + animation frames (if created)")
    print("\n" + "="*70)