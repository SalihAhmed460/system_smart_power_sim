"""
Dashboard interface for the Power Smart System.
Provides system monitoring, simulation controls, and visualization in a single interface.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from typing import Dict, List, Optional
import json
import datetime

from nuclear_simulator import NuclearSimulator
from ml_model import NeutronPredictionModel, TENSORFLOW_AVAILABLE

class SimulationHistory:
    """Manages the history of simulations for comparison and analysis."""
    
    def __init__(self, history_file="simulation_history.json"):
        self.history_file = history_file
        self.history = self._load_history()
    
    def _load_history(self):
        """Load simulation history from file if it exists."""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            else:
                return []
        except Exception as e:
            print(f"Error loading history: {e}")
            return []
    
    def save_simulation(self, params, result_data, prediction_data=None):
        """Save a simulation run to history."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate some statistics about the simulation
        final_neutrons = result_data[-1] if result_data else 0
        neutron_growth = None
        if len(result_data) > 1:
            neutron_growth = final_neutrons / max(1, result_data[0])
        
        # Create the history entry
        entry = {
            "timestamp": timestamp,
            "parameters": params,
            "results": {
                "generations": len(result_data) - 1,  # Subtract 1 because first entry is initial count
                "initial_neutrons": result_data[0] if result_data else 0,
                "final_neutrons": final_neutrons,
                "neutron_growth_factor": neutron_growth,
                "prediction_accuracy": self._calc_prediction_accuracy(prediction_data) if prediction_data else None
            },
            "data": {
                "neutron_counts": result_data,
                "predictions": prediction_data
            }
        }
        
        self.history.append(entry)
        
        # Only keep recent history (last 20 entries)
        if len(self.history) > 20:
            self.history = self.history[-20:]
        
        self._save_to_file()
        return entry
    
    def _calc_prediction_accuracy(self, predictions):
        """Calculate the accuracy of predictions if real values are available."""
        # This would require comparing with real data which we don't have for future predictions
        # For now, return a placeholder
        return {"accuracy_score": None}
    
    def _save_to_file(self):
        """Save history to file."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f)
        except Exception as e:
            print(f"Error saving history: {e}")
    
    def get_recent_simulations(self, limit=5):
        """Get most recent simulations."""
        return self.history[-limit:] if self.history else []

class Dashboard:
    def __init__(self, root):
        """Initialize the dashboard interface."""
        self.root = root
        self.root.title("Power Smart System Dashboard")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 600)
        
        # Setup resources
        self.simulator = NuclearSimulator()
        self.ml_model = NeutronPredictionModel()
        self.history_manager = SimulationHistory()
        
        # Data storage
        self.current_simulation = []
        self.current_predictions = []
        self.simulation_running = False
        self.ai_training = False
        
        # Create UI
        self._setup_ui()
        
        # Load recent history
        self._refresh_history_display()
    
    def _setup_ui(self):
        """Setup the dashboard user interface."""
        # Create main frame structure
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create splitter layout - left panel and right panel
        self.paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel: Controls, Parameters, System Info
        left_panel = ttk.Frame(self.paned_window, width=300)
        self.paned_window.add(left_panel, weight=1)
        
        # Right panel: Visualization and output
        right_panel = ttk.Frame(self.paned_window)
        self.paned_window.add(right_panel, weight=3)
        
        # Configure left panel
        left_panel.grid_columnconfigure(0, weight=1)
        
        # Status indicator frame
        status_frame = ttk.LabelFrame(left_panel, text="System Status")
        status_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        # Status indicators
        self.status_indicators = {}
        
        # System status
        system_status_frame = ttk.Frame(status_frame)
        system_status_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(system_status_frame, text="System Status:").pack(side=tk.LEFT)
        status_label = ttk.Label(system_status_frame, text="Ready", foreground="green")
        status_label.pack(side=tk.RIGHT)
        self.status_indicators["system"] = status_label
        
        # AI model status
        ai_status_frame = ttk.Frame(status_frame)
        ai_status_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(ai_status_frame, text="AI Model:").pack(side=tk.LEFT)
        ai_status_text = "TensorFlow Active" if TENSORFLOW_AVAILABLE else "Using Statistical Model"
        ai_status_color = "green" if TENSORFLOW_AVAILABLE else "orange"
        ai_status = ttk.Label(ai_status_frame, text=ai_status_text, foreground=ai_status_color)
        ai_status.pack(side=tk.RIGHT)
        self.status_indicators["ai_model"] = ai_status
        
        # Simulation status
        sim_status_frame = ttk.Frame(status_frame)
        sim_status_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(sim_status_frame, text="Simulation:").pack(side=tk.LEFT)
        sim_status = ttk.Label(sim_status_frame, text="Idle", foreground="blue")
        sim_status.pack(side=tk.RIGHT)
        self.status_indicators["simulation"] = sim_status
        
        # Parameter configuration frame
        param_frame = ttk.LabelFrame(left_panel, text="Simulation Parameters")
        param_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        # Parameter inputs
        ttk.Label(param_frame, text="Generations:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.gen_var = tk.StringVar(value="10")
        ttk.Entry(param_frame, textvariable=self.gen_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(param_frame, text="Initial Neutrons:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.init_var = tk.StringVar(value="1")
        ttk.Entry(param_frame, textvariable=self.init_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(param_frame, text="Fission Probability:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.prob_var = tk.StringVar(value="0.7")
        ttk.Entry(param_frame, textvariable=self.prob_var, width=10).grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(param_frame, text="Neutrons per Fission:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.nfission_var = tk.StringVar(value="3")
        ttk.Entry(param_frame, textvariable=self.nfission_var, width=10).grid(row=3, column=1, padx=5, pady=5)
        
        ttk.Label(param_frame, text="Prediction Generations:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.pred_var = tk.StringVar(value="5")
        ttk.Entry(param_frame, textvariable=self.pred_var, width=10).grid(row=4, column=1, padx=5, pady=5)
        
        # Control buttons frame
        control_frame = ttk.LabelFrame(left_panel, text="Controls")
        control_frame.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        
        # Control buttons
        self.run_btn = ttk.Button(control_frame, text="Run Simulation", command=self.run_simulation)
        self.run_btn.pack(fill=tk.X, padx=5, pady=5)
        
        self.predict_btn = ttk.Button(control_frame, text="Generate AI Prediction", 
                                     command=self.generate_prediction, state="disabled")
        self.predict_btn.pack(fill=tk.X, padx=5, pady=5)
        
        self.save_btn = ttk.Button(control_frame, text="Save Results", 
                                  command=self.save_results, state="disabled")
        self.save_btn.pack(fill=tk.X, padx=5, pady=5)
        
        self.clear_btn = ttk.Button(control_frame, text="Clear", command=self.clear_results)
        self.clear_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # History section
        history_frame = ttk.LabelFrame(left_panel, text="Simulation History")
        history_frame.grid(row=3, column=0, padx=5, pady=5, sticky="nsew")
        left_panel.grid_rowconfigure(3, weight=1)
        
        # History list
        self.history_list = ttk.Treeview(history_frame, columns=('date', 'neutrons'), show='headings', height=6)
        self.history_list.heading('date', text='Date')
        self.history_list.heading('neutrons', text='Final Neutrons')
        self.history_list.column('date', width=120)
        self.history_list.column('neutrons', width=80)
        self.history_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.history_list.bind("<Double-1>", self.on_history_selected)
        
        # Right panel configuration
        right_panel.grid_columnconfigure(0, weight=1)
        right_panel.grid_rowconfigure(0, weight=3)
        right_panel.grid_rowconfigure(1, weight=1)
        
        # Visualization frame
        viz_frame = ttk.LabelFrame(right_panel, text="Visualization")
        viz_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar_frame = ttk.Frame(viz_frame)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        
        # Output frame
        output_frame = ttk.LabelFrame(right_panel, text="Simulation Output")
        output_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        
        # Output text
        self.output_text = scrolledtext.ScrolledText(output_frame, height=10)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add initial status message
        self.log_output("Dashboard initialized. Ready to run simulations.")
        self.log_output(f"AI Model Status: {'TensorFlow available' if TENSORFLOW_AVAILABLE else 'Using statistical model (TensorFlow not available)'}")
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(main_frame, orient='horizontal', mode='indeterminate')
        self.progress_bar.pack(fill=tk.X, padx=10, pady=(0, 10))
        
    def log_output(self, message):
        """Add a message to the output text widget."""
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.output_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.output_text.see(tk.END)  # Scroll to the end
    
    def validate_inputs(self):
        """Validate user inputs and return them as appropriate types."""
        try:
            generations = int(self.gen_var.get())
            initial_neutrons = int(self.init_var.get())
            fission_prob = float(self.prob_var.get())
            neutrons_per_fission = float(self.nfission_var.get())
            pred_gens = int(self.pred_var.get())
            
            if generations <= 0 or initial_neutrons <= 0 or pred_gens < 0:
                raise ValueError("Values must be positive")
            if not 0 <= fission_prob <= 1:
                raise ValueError("Fission probability must be between 0 and 1")
                
            return {
                "generations": generations,
                "initial_neutrons": initial_neutrons,
                "fission_prob": fission_prob,
                "neutrons_per_fission": neutrons_per_fission,
                "pred_gens": pred_gens
            }
            
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please check your inputs: {str(e)}")
            return None
    
    def run_simulation(self):
        """Run the nuclear simulation with the specified parameters."""
        params = self.validate_inputs()
        if not params:
            return
        
        # Update status
        self.simulation_running = True
        self.status_indicators["simulation"].config(text="Running", foreground="orange")
        self.status_indicators["system"].config(text="Busy", foreground="orange")
        self.run_btn.config(state="disabled")
        self.predict_btn.config(state="disabled")
        self.save_btn.config(state="disabled")
        self.progress_bar.start()
        
        # Clear previous results
        self.current_simulation = []
        self.current_predictions = []
        self.log_output("Starting new simulation...")
        
        # Run simulation in a separate thread
        def simulation_thread():
            try:
                # Configure simulator
                self.simulator = NuclearSimulator(
                    fission_probability=params["fission_prob"],
                    neutrons_per_fission=params["neutrons_per_fission"]
                )
                
                # Run simulation
                simulation_data = self.simulator.run_simulation(
                    initial_neutrons=params["initial_neutrons"],
                    generations=params["generations"]
                )
                
                self.current_simulation = simulation_data
                
                # Update UI in the main thread
                self.root.after(0, self.display_simulation_results)
                
            except Exception as e:
                self.root.after(0, lambda: self.handle_error(f"Simulation error: {str(e)}"))
                
        threading.Thread(target=simulation_thread).start()
    
    def display_simulation_results(self):
        """Display the simulation results."""
        self.simulation_running = False
        self.progress_bar.stop()
        self.status_indicators["simulation"].config(text="Complete", foreground="green")
        self.status_indicators["system"].config(text="Ready", foreground="green")
        self.run_btn.config(state="normal")
        self.predict_btn.config(state="normal")
        self.save_btn.config(state="normal")
        
        # Update visualization
        self.plot_results()
        
        # Log results
        final_neutrons = self.current_simulation[-1] if self.current_simulation else 0
        self.log_output(f"Simulation complete. Final neutron count: {final_neutrons}")
    
    def generate_prediction(self):
        """Generate AI prediction for future neutron counts."""
        if not self.current_simulation:
            self.log_output("Error: No simulation data available for prediction.")
            return
        
        params = self.validate_inputs()
        if not params:
            return
        
        # Update status
        self.ai_training = True
        self.status_indicators["system"].config(text="Training", foreground="orange")
        model_type = "statistical model" if not TENSORFLOW_AVAILABLE else "neural network"
        self.status_indicators["ai_model"].config(text=f"Training {model_type}", foreground="orange")
        self.predict_btn.config(state="disabled")
        self.progress_bar.start()
        
        self.log_output(f"Training {model_type} for prediction...")
        
        # Run AI prediction in a separate thread
        def prediction_thread():
            try:
                # Get simulation data
                sim_data = self.simulator.get_simulation_data()
                
                # Train the model
                self.ml_model.train(sim_data, epochs=50, batch_size=32)
                
                # Make predictions
                predictions = self.ml_model.predict_next_generations(
                    sim_data, num_predictions=params["pred_gens"])
                
                self.current_predictions = predictions
                
                # Update UI in the main thread
                self.root.after(0, self.display_prediction_results)
                
            except Exception as e:
                self.root.after(0, lambda: self.handle_error(f"Prediction error: {str(e)}"))
                
        threading.Thread(target=prediction_thread).start()
    
    def display_prediction_results(self):
        """Display the prediction results."""
        self.ai_training = False
        self.progress_bar.stop()
        self.status_indicators["system"].config(text="Ready", foreground="green")
        model_type = "Statistical Model" if not TENSORFLOW_AVAILABLE else "TensorFlow Active"
        status_color = "orange" if not TENSORFLOW_AVAILABLE else "green"
        self.status_indicators["ai_model"].config(text=model_type, foreground=status_color)
        self.predict_btn.config(state="normal")
        
        # Update visualization
        self.plot_results()
        
        # Log results
        final_prediction = self.current_predictions[-1] if self.current_predictions else 0
        self.log_output(f"Prediction complete. Final predicted neutron count: {final_prediction}")
    
    def plot_results(self):
        """Plot the simulation and prediction results."""
        self.ax.clear()
        
        # Plot actual data if available
        if self.current_simulation:
            generations = range(len(self.current_simulation))
            self.ax.plot(generations, self.current_simulation, 'b-', label='Actual', marker='o')
            
            # Plot predictions if available
            if self.current_predictions:
                pred_generations = range(len(self.current_simulation), 
                                        len(self.current_simulation) + len(self.current_predictions))
                self.ax.plot(pred_generations, self.current_predictions, 'r--', label='Predicted', marker='x')
                
                # Add connecting line
                self.ax.plot([len(self.current_simulation)-1, len(self.current_simulation)], 
                            [self.current_simulation[-1], self.current_predictions[0]], 
                            'k:', alpha=0.5)
                
            # Configure plot
            self.ax.set_title('Nuclear Chain Reaction Simulation')
            self.ax.set_xlabel('Generation')
            self.ax.set_ylabel('Number of Neutrons')
            self.ax.grid(True, linestyle='--', alpha=0.7)
            self.ax.legend()
            
            # Use log scale if values get very large
            max_value = max(self.current_simulation + self.current_predictions)
            if max_value > 1000:
                self.ax.set_yscale('log')
                self.log_output("Using logarithmic scale due to large neutron counts.")
            else:
                self.ax.set_yscale('linear')
                
        # Refresh the canvas
        self.canvas.draw()
    
    def save_results(self):
        """Save the current simulation and prediction results."""
        if not self.current_simulation:
            self.log_output("Nothing to save. Run a simulation first.")
            return
        
        params = self.validate_inputs()
        if not params:
            return
        
        # Save to history
        entry = self.history_manager.save_simulation(params, self.current_simulation, self.current_predictions)
        
        # Update history display
        self._refresh_history_display()
        
        # Log
        self.log_output(f"Saved simulation from {entry['timestamp']} to history.")
    
    def _refresh_history_display(self):
        """Refresh the history list display."""
        # Clear existing items
        for item in self.history_list.get_children():
            self.history_list.delete(item)
        
        # Load recent history
        recent_sims = self.history_manager.get_recent_simulations()
        
        # Add to list in reverse order (newest first)
        for sim in reversed(recent_sims):
            self.history_list.insert('', 'end', values=(
                sim["timestamp"], 
                sim["results"]["final_neutrons"]
            ))
    
    def on_history_selected(self, event):
        """Handle history item selection."""
        try:
            selection = self.history_list.selection()[0]
            item_values = self.history_list.item(selection, "values")
            timestamp = item_values[0]
            
            # Find the entry in history
            for entry in self.history_manager.history:
                if entry["timestamp"] == timestamp:
                    # Load the data
                    self.current_simulation = entry["data"]["neutron_counts"]
                    self.current_predictions = entry["data"]["predictions"] or []
                    
                    # Update parameters in the UI
                    params = entry["parameters"]
                    self.gen_var.set(str(params["generations"]))
                    self.init_var.set(str(params["initial_neutrons"]))
                    self.prob_var.set(str(params["fission_prob"]))
                    self.nfission_var.set(str(params["neutrons_per_fission"]))
                    self.pred_var.set(str(params["pred_gens"]))
                    
                    # Update plot
                    self.plot_results()
                    
                    # Enable save and predict buttons
                    self.save_btn.config(state="normal")
                    self.predict_btn.config(state="normal")
                    
                    self.log_output(f"Loaded simulation from {timestamp}")
                    break
        except Exception as e:
            self.log_output(f"Error loading history: {e}")
    
    def clear_results(self):
        """Clear current results."""
        self.current_simulation = []
        self.current_predictions = []
        self.ax.clear()
        self.ax.set_title('Nuclear Chain Reaction Simulation')
        self.ax.set_xlabel('Generation')
        self.ax.set_ylabel('Number of Neutrons')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.canvas.draw()
        self.predict_btn.config(state="disabled")
        self.save_btn.config(state="disabled")
        self.log_output("Cleared current results.")
    
    def handle_error(self, error_message):
        """Handle exceptions that occur during simulation or prediction."""
        self.progress_bar.stop()
        self.simulation_running = False
        self.ai_training = False
        self.status_indicators["system"].config(text="Error", foreground="red")
        self.status_indicators["simulation"].config(text="Error", foreground="red")
        self.run_btn.config(state="normal")
        self.predict_btn.config(state="disabled")
        self.log_output(f"ERROR: {error_message}")
        messagebox.showerror("Error", error_message)

def run_dashboard():
    """Run the dashboard application."""
    root = tk.Tk()
    app = Dashboard(root)
    root.mainloop()

if __name__ == "__main__":
    run_dashboard()
