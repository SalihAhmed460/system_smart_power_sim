"""
Performance analysis tool for the Power Smart System.
This script runs benchmarks on various components and generates performance metrics.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os
import platform
import psutil  # Fixed typo from 'psutila'
import gc

# Import system components
from nuclear_simulator import NuclearSimulator
from ml_model import NeutronPredictionModel, TENSORFLOW_AVAILABLE

class PerformanceAnalyzer:
    def __init__(self, output_dir: str = "./performance_results"):
        """Initialize the performance analyzer."""
        self.output_dir = output_dir
        self.results = {}
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Record system info
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict[str, str]:
        """Get information about the system running the tests."""
        info = {
            "Platform": platform.platform(),
            "Python Version": platform.python_version(),
            "Processor": platform.processor(),
            "Machine": platform.machine(),
            "CPU Count": str(psutil.cpu_count(logical=False)),
            "Logical CPUs": str(psutil.cpu_count(logical=True)),
            "TensorFlow Available": str(TENSORFLOW_AVAILABLE)
        }
        
        # Add memory information
        memory = psutil.virtual_memory()
        info["Total Memory"] = f"{memory.total / (1024**3):.2f} GB"
        info["Available Memory"] = f"{memory.available / (1024**3):.2f} GB"
        
        return info
    
    def benchmark_simulator(self, 
                          generations_range: List[int] = [10, 50, 100, 500, 1000],
                          runs_per_test: int = 3) -> Dict:
        """Benchmark the nuclear simulator with various generation counts."""
        print("Benchmarking Nuclear Simulator...")
        results = {"generations": [], "time_seconds": [], "memory_mb": []}
        
        for generations in generations_range:
            gen_times = []
            gen_memory = []
            
            for _ in range(runs_per_test):
                # Force garbage collection to get more accurate memory measurement
                gc.collect()
                initial_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                
                # Time the simulation
                simulator = NuclearSimulator(fission_probability=0.7, neutrons_per_fission=3)
                start_time = time.time()
                simulator.run_simulation(initial_neutrons=5, generations=generations)
                end_time = time.time()
                
                # Calculate memory usage after simulation
                final_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                memory_used = final_memory - initial_memory
                
                gen_times.append(end_time - start_time)
                gen_memory.append(memory_used)
            
            # Record average results
            results["generations"].append(generations)
            results["time_seconds"].append(np.mean(gen_times))
            results["memory_mb"].append(np.mean(gen_memory))
            
            print(f"  Generations: {generations}, Avg. Time: {np.mean(gen_times):.4f}s, Avg. Memory: {np.mean(gen_memory):.2f}MB")
        
        self.results["simulator"] = results
        return results
    
    def benchmark_ml_model(self,
                         sequence_length_range: List[int] = [3, 5, 10, 15],
                         data_points: int = 100,
                         runs_per_test: int = 3) -> Dict:
        """Benchmark the ML model with various sequence lengths."""
        print("Benchmarking ML Model...")
        results = {"sequence_length": [], "training_time": [], "prediction_time": [], "using_tensorflow": TENSORFLOW_AVAILABLE}
        
        # Generate some consistent test data
        simulator = NuclearSimulator(fission_probability=0.7, neutrons_per_fission=3)
        sim_data = simulator.run_simulation(initial_neutrons=5, generations=data_points)
        sim_dict = simulator.get_simulation_data()
        
        for seq_len in sequence_length_range:
            training_times = []
            prediction_times = []
            
            for _ in range(runs_per_test):
                # Create and train model
                model = NeutronPredictionModel(sequence_length=seq_len)
                
                start_time = time.time()
                model.train(sim_dict, epochs=20, batch_size=4)
                training_end_time = time.time()
                
                # Make predictions
                model.predict_next_generations(sim_dict, num_predictions=10)
                prediction_end_time = time.time()
                
                training_times.append(training_end_time - start_time)
                prediction_times.append(prediction_end_time - training_end_time)
            
            # Record average results
            results["sequence_length"].append(seq_len)
            results["training_time"].append(np.mean(training_times))
            results["prediction_time"].append(np.mean(prediction_times))
            
            print(f"  Sequence Length: {seq_len}, Training: {np.mean(training_times):.4f}s, Prediction: {np.mean(prediction_times):.4f}s")
        
        self.results["ml_model"] = results
        return results
    
    def plot_results(self):
        """Plot the benchmark results."""
        if not self.results:
            print("No benchmark results to plot.")
            return
        
        # Create figure with system information
        plt.figure(figsize=(14, 10))
        plt.suptitle("Power Smart System Performance Analysis", fontsize=16)
        
        # Add system info as text
        system_info_str = "\n".join([f"{k}: {v}" for k, v in self.system_info.items()])
        plt.figtext(0.02, 0.02, f"System Information:\n{system_info_str}", wrap=True, fontsize=9)
        
        # Plot simulator performance
        if "simulator" in self.results:
            ax1 = plt.subplot(2, 2, 1)
            sim_data = self.results["simulator"]
            ax1.plot(sim_data["generations"], sim_data["time_seconds"], 'o-', label='Execution Time')
            ax1.set_xlabel('Number of Generations')
            ax1.set_ylabel('Time (seconds)')
            ax1.set_title('Simulation Execution Time')
            ax1.grid(True, alpha=0.3)
            
            ax2 = plt.subplot(2, 2, 2)
            ax2.plot(sim_data["generations"], sim_data["memory_mb"], 'o-', color='green', label='Memory Usage')
            ax2.set_xlabel('Number of Generations')
            ax2.set_ylabel('Memory Usage (MB)')
            ax2.set_title('Simulation Memory Usage')
            ax2.grid(True, alpha=0.3)
        
        # Plot ML model performance
        if "ml_model" in self.results:
            ml_data = self.results["ml_model"]
            
            ax3 = plt.subplot(2, 2, 3)
            ax3.plot(ml_data["sequence_length"], ml_data["training_time"], 'o-', color='red', label='Training Time')
            ax3.set_xlabel('Sequence Length')
            ax3.set_ylabel('Time (seconds)')
            ax3.set_title(f'ML Model Training Time ({"TensorFlow" if ml_data["using_tensorflow"] else "Statistical Model"})')
            ax3.grid(True, alpha=0.3)
            
            ax4 = plt.subplot(2, 2, 4)
            ax4.plot(ml_data["sequence_length"], ml_data["prediction_time"], 'o-', color='purple', label='Prediction Time')
            ax4.set_xlabel('Sequence Length')
            ax4.set_ylabel('Time (seconds)')
            ax4.set_title('ML Model Prediction Time')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.savefig(os.path.join(self.output_dir, 'performance_metrics.png'), dpi=150)
        plt.close()
        
        print(f"Performance plots saved to {os.path.join(self.output_dir, 'performance_metrics.png')}")
    
    def export_results(self):
        """Export the benchmark results to a text file."""
        if not self.results:
            print("No benchmark results to export.")
            return
            
        filepath = os.path.join(self.output_dir, 'performance_data.txt')
        
        with open(filepath, 'w') as f:
            # Write header
            f.write("======== Power Smart System Performance Report ========\n\n")
            
            # Write system information
            f.write("System Information:\n")
            for key, value in self.system_info.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Write simulator results
            if "simulator" in self.results:
                f.write("Nuclear Simulator Performance:\n")
                f.write("Generations | Time (s) | Memory (MB)\n")
                f.write("----------------------------------\n")
                
                sim_data = self.results["simulator"]
                for i in range(len(sim_data["generations"])):
                    f.write(f"{sim_data['generations'][i]:11d} | {sim_data['time_seconds'][i]:8.4f} | {sim_data['memory_mb'][i]:10.2f}\n")
                f.write("\n")
            
            # Write ML model results
            if "ml_model" in self.results:
                ml_data = self.results["ml_model"]
                model_type = "TensorFlow" if ml_data["using_tensorflow"] else "Statistical Model"
                
                f.write(f"ML Model Performance ({model_type}):\n")
                f.write("Sequence Length | Training Time (s) | Prediction Time (s)\n")
                f.write("---------------------------------------------------\n")
                
                for i in range(len(ml_data["sequence_length"])):
                    f.write(f"{ml_data['sequence_length'][i]:15d} | {ml_data['training_time'][i]:16.4f} | {ml_data['prediction_time'][i]:18.4f}\n")
                f.write("\n")
                
        print(f"Performance data exported to {filepath}")
        

def run_comprehensive_analysis():
    """Run a comprehensive analysis of the system performance."""
    analyzer = PerformanceAnalyzer()
    
    print("=" * 50)
    print("Power Smart System Performance Analysis")
    print("=" * 50)
    print("\nSystem Information:")
    for key, value in analyzer.system_info.items():
        print(f"  {key}: {value}")
    print("\nRunning benchmarks...")
    
    # Run benchmarks with appropriate parameters based on available resources
    # Adjust these for different systems
    memory = psutil.virtual_memory()
    if memory.total > 8 * (1024**3):  # More than 8GB RAM
        generations_range = [10, 50, 100, 250, 500, 1000]
        sequence_lengths = [3, 5, 10, 20]
    else:  # Limited memory
        generations_range = [10, 25, 50, 100, 200]
        sequence_lengths = [3, 5, 8]
    
    analyzer.benchmark_simulator(generations_range=generations_range)
    analyzer.benchmark_ml_model(sequence_length_range=sequence_lengths)
    
    analyzer.plot_results()
    analyzer.export_results()
    
    print("\nAnalysis complete!")
    print("Check the 'performance_results' directory for output files.")

if __name__ == "__main__":
    run_comprehensive_analysis()
