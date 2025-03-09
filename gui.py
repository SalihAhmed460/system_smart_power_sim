import tkinter as tk
from tkinter import ttk, messagebox, Frame
import threading
import time
from typing import Callable
import warnings

from nuclear_simulator import NuclearSimulator
from ml_model import NeutronPredictionModel

# Check if TensorFlow is available
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Import visualization module
try:
    from visualization import NuclearVisualizer
except ImportError as e:
    warnings.warn(f"Error importing visualization: {e}")
    # Create a simplified visualization class
    class NuclearVisualizer:
        def __init__(self, master=None):
            self.master = master
            self.widget = tk.Label(master, text="Visualization not available")
        
        def setup_plot(self, figsize=None):
            pass
        
        def plot_simulation(self, actual_data, predictions=None):
            pass

class NuclearSimulationApp:
    def __init__(self, root):
        """
        Initialize the GUI application.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("AI-Powered Nuclear Energy Simulation")
        self.root.geometry("1200x800")
        self.root.configure(bg="#e0e0e0")
        
        style = ttk.Style()
        style.configure("TLabel", background="#e0e0e0", foreground="#333333", font=("Segoe UI", 11, "bold"))
        style.configure("TFrame", background="#e0e0e0")
        style.configure("TButton", background="#007acc", foreground="#ffffff", font=("Segoe UI", 10, "bold"))
        style.configure("TLabelFrame", background="#e0e0e0", borderwidth=2, relief="groove")
        style.configure("TEntry", fieldbackground="#ffffff", background="#e0e0e0", font=("Segoe UI", 10))
        style.configure("TProgressbar", background="#007acc")
        
        self.simulator = NuclearSimulator()
        self.ml_model = NeutronPredictionModel()
        
        # Check if running with fallback mode
        self.using_fallback = not TENSORFLOW_AVAILABLE
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Set up the user interface components."""
        # Create frames
        self.control_frame = ttk.LabelFrame(self.root, text="Simulation Controls")
        self.control_frame.pack(padx=10, pady=10, fill="x")
        
        self.viz_frame = ttk.LabelFrame(self.root, text="Visualization")
        self.viz_frame.pack(padx=10, pady=5, fill="both", expand=True)
        
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(padx=10, pady=5, fill="x")
        
        # Control widgets
        ttk.Label(self.control_frame, text="Generations:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.gen_var = tk.StringVar(value="10")
        ttk.Entry(self.control_frame, textvariable=self.gen_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(self.control_frame, text="Initial Neutrons:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.init_var = tk.StringVar(value="1")
        ttk.Entry(self.control_frame, textvariable=self.init_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(self.control_frame, text="Fission Probability:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.prob_var = tk.StringVar(value="0.7")
        ttk.Entry(self.control_frame, textvariable=self.prob_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(self.control_frame, text="Neutrons per Fission:").grid(row=1, column=2, padx=5, pady=5, sticky="w")
        self.nfission_var = tk.StringVar(value="3")
        ttk.Entry(self.control_frame, textvariable=self.nfission_var, width=10).grid(row=1, column=3, padx=5, pady=5)
        
        ttk.Label(self.control_frame, text="AI Prediction Generations:").grid(row=1, column=4, padx=5, pady=5, sticky="w")
        self.pred_var = tk.StringVar(value="5")
        ttk.Entry(self.control_frame, textvariable=self.pred_var, width=10).grid(row=1, column=5, padx=5, pady=5)
        
        # Buttons
        buttons_frame = ttk.Frame(self.control_frame)
        buttons_frame.grid(row=0, column=4, rowspan=1, columnspan=2, padx=5, pady=5)
        
        self.run_btn = ttk.Button(buttons_frame, text="Run Simulation", command=self.run_simulation)
        self.run_btn.pack(side="left", padx=5)
        
        self.predict_btn = ttk.Button(buttons_frame, text="Predict with AI", command=self.predict_with_ai, state="disabled")
        self.predict_btn.pack(side="right", padx=5)
        
        # Add indicator for AI mode
        ai_mode_text = "Using Simple Statistics Model" if self.using_fallback else "Using TensorFlow AI Model"
        ai_mode_color = "orange" if self.using_fallback else "green"
        
        ai_mode_label = ttk.Label(
            self.control_frame, 
            text=ai_mode_text,
            foreground=ai_mode_color,
            font=("Arial", 10, "italic")
        )
        ai_mode_label.grid(row=2, column=0, columnspan=6, padx=5, pady=(10,0))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.progress_bar = ttk.Progressbar(self.status_frame, orient="horizontal", mode="indeterminate")
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        ttk.Label(self.status_frame, textvariable=self.status_var).pack(side="right", padx=5)
        
        # Visualization
        self.visualizer = NuclearVisualizer(self.viz_frame)
        self.visualizer.setup_plot()
        self.visualizer.widget.pack(fill="both", expand=True)
        
        # Data storage
        self.simulation_data = []
        self.predictions = []

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
                
            return generations, initial_neutrons, fission_prob, neutrons_per_fission, pred_gens
            
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please check your inputs: {str(e)}")
            return None
            
    def run_simulation(self):
        """Run the nuclear simulation with the specified parameters."""
        params = self.validate_inputs()
        if not params:
            return
            
        generations, initial_neutrons, fission_prob, neutrons_per_fission, _ = params
        
        # Update UI
        self.run_btn.config(state="disabled")
        self.predict_btn.config(state="disabled")
        self.status_var.set("Running simulation...")
        self.progress_bar.start()
        
        # Run simulation in a separate thread
        def simulation_thread():
            try:
                # Configure simulator
                self.simulator = NuclearSimulator(
                    fission_probability=fission_prob,
                    neutrons_per_fission=neutrons_per_fission
                )
                
                # Run simulation
                self.simulation_data = self.simulator.run_simulation(
                    initial_neutrons=initial_neutrons,
                    generations=generations
                )
                
                # Update UI in the main thread
                self.root.after(0, self.display_results)
                
            except Exception as e:
                self.root.after(0, lambda: self.handle_error(str(e)))
                
        threading.Thread(target=simulation_thread).start()
        
    def display_results(self):
        """Display the simulation results."""
        self.progress_bar.stop()
        self.status_var.set("Simulation complete")
        self.run_btn.config(state="normal")
        self.predict_btn.config(state="normal")
        
        # Update the visualization
        self.visualizer.plot_simulation(self.simulation_data, self.predictions)
        
    def predict_with_ai(self):
        """Use the ML model to predict future neutron counts."""
        if not self.simulation_data:
            messagebox.showerror("Error", "Please run a simulation first.")
            return
            
        try:
            pred_gens = int(self.pred_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number for prediction generations.")
            return
            
        # Update UI
        self.predict_btn.config(state="disabled")
        model_type = "statistical model" if self.using_fallback else "AI model"
        self.status_var.set(f"Training {model_type}...")
        self.progress_bar.start()
        
        # Run AI prediction in a separate thread
        def prediction_thread():
            try:
                # Get simulation data in the right format
                sim_data = self.simulator.get_simulation_data()
                
                # Train the model
                self.ml_model.train(sim_data, epochs=50, batch_size=32)
                
                # Make predictions
                self.predictions = self.ml_model.predict_next_generations(sim_data, num_predictions=pred_gens)
                
                # Update UI in the main thread
                self.root.after(0, self.display_predictions)
                
            except Exception as e:
                self.root.after(0, lambda: self.handle_error(str(e)))
                
        threading.Thread(target=prediction_thread).start()
        
    def display_predictions(self):
        """Display the AI predictions."""
        self.progress_bar.stop()
        model_type = "statistical" if self.using_fallback else "AI"
        self.status_var.set(f"{model_type.capitalize()} prediction complete")
        self.predict_btn.config(state="normal")
        
        # Update the visualization
        self.visualizer.plot_simulation(self.simulation_data, self.predictions)
        
    def handle_error(self, error_msg):
        """Handle exceptions that occur during simulation or prediction."""
        self.progress_bar.stop()
        self.status_var.set("Error")
        self.run_btn.config(state="normal")
        self.predict_btn.config(state="normal")
        messagebox.showerror("Error", f"An error occurred: {error_msg}")
