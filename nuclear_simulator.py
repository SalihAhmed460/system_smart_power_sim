import numpy as np
from typing import List, Tuple, Dict

class NuclearSimulator:
    def __init__(self, fission_probability: float = 0.7, neutrons_per_fission: int = 3):
        """
        Initialize the nuclear simulator.
        
        Args:
            fission_probability: Probability of a neutron causing fission
            neutrons_per_fission: Average number of neutrons released per fission
        """
        self.fission_probability = fission_probability
        self.neutrons_per_fission = neutrons_per_fission
        self.history = []
        
    def run_simulation(self, initial_neutrons: int = 1, generations: int = 10) -> List[int]:
        """
        Run the nuclear chain reaction simulation.
        
        Args:
            initial_neutrons: Number of neutrons to start the simulation with
            generations: Number of generations to simulate
            
        Returns:
            List of neutron counts for each generation
        """
        neutron_counts = [initial_neutrons]
        current_neutrons = initial_neutrons
        
        for gen in range(generations):
            # Each neutron has a chance to cause fission
            fissions = np.random.binomial(current_neutrons, self.fission_probability)
            
            # Each fission produces new neutrons
            new_neutrons = np.random.poisson(self.neutrons_per_fission, fissions).sum()
            
            # Store the result
            neutron_counts.append(new_neutrons)
            current_neutrons = new_neutrons
            
        self.history = neutron_counts
        return neutron_counts
    
    def get_simulation_data(self) -> Dict:
        """
        Get the simulation data in a format suitable for ML processing.
        """
        return {
            "neutron_counts": self.history,
            "fission_probability": self.fission_probability,
            "neutrons_per_fission": self.neutrons_per_fission
        }
