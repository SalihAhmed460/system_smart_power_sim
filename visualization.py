import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from typing import List, Optional

class NuclearVisualizer:
    def __init__(self, master=None):
        """
        Initialize the visualization module.
        
        Args:
            master: Tkinter master window for embedding plots
        """
        self.master = master
        self.figure = None
        self.canvas = None
        
    def setup_plot(self, figsize=(10, 6)):
        """Set up the matplotlib figure."""
        self.figure = Figure(figsize=figsize, dpi=100)
        self.ax = self.figure.add_subplot(111)
        
        if self.master:
            self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
            self.widget = self.canvas.get_tk_widget()
            
    def plot_simulation(self, actual_data: List[int], predictions: Optional[List[int]] = None):
        """
        Plot the simulation results.
        
        Args:
            actual_data: List of actual neutron counts
            predictions: Optional list of predicted neutron counts
        """
        if self.figure is None:
            self.setup_plot()
            
        self.ax.clear()
        
        # Plot actual data
        generations = range(len(actual_data))
        self.ax.plot(generations, actual_data, 'b-', label='Actual', marker='o')
        
        # Plot predictions if available
        if predictions and len(predictions) > 0:
            pred_generations = range(len(actual_data), len(actual_data) + len(predictions))
            self.ax.plot(pred_generations, predictions, 'r--', label='Predicted', marker='x')
            
            # Add connecting line between actual and predicted
            self.ax.plot([len(actual_data)-1, len(actual_data)], 
                         [actual_data[-1], predictions[0]], 
                         'k:', alpha=0.5)
        
        # Configure plot
        self.ax.set_title('Nuclear Chain Reaction Simulation')
        self.ax.set_xlabel('Generation')
        self.ax.set_ylabel('Number of Neutrons')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.legend()
        
        # Use log scale if values get very large
        if max(actual_data + (predictions or [])) > 1000:
            self.ax.set_yscale('log')
            
        # Refresh the canvas
        if self.canvas:
            self.canvas.draw()
            
    def save_plot(self, filename='nuclear_simulation.png'):
        """Save the current plot to a file."""
        self.figure.savefig(filename)
