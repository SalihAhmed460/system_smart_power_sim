import numpy as np
import os
import warnings
import sys
import platform
from typing import List, Tuple, Dict, Optional, Any

# Configure warnings to be more concise
warnings.filterwarnings('always', category=UserWarning)

# Try to import TensorFlow, but provide multiple fallbacks
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
    USING_COMPATIBILITY_LAYER = False
    print(f"Using TensorFlow {tf.__version__} for neural network predictions")
except ImportError as e:
    # Try the compatibility layer
    try:
        import tensorflow_compat as tf
        from tensorflow_compat.keras.models import Sequential
        from tensorflow_compat.keras.layers import Dense, LSTM, Dropout
        from tensorflow_compat.keras.optimizers import Adam
        TENSORFLOW_AVAILABLE = True
        USING_COMPATIBILITY_LAYER = True
        print(f"Using TensorFlow compatibility layer ({tf.__version__}) with scikit-learn backend")
    except ImportError:
        # Provide more concise warning
        warning_message = "TensorFlow not available. Run 'python install_tensorflow.py' for AI features."
        warnings.warn(warning_message)
        TENSORFLOW_AVAILABLE = False
        USING_COMPATIBILITY_LAYER = False
        print("Using simplified statistical model for predictions")
        
        # Define dummy Sequential class to avoid NameError
        class Sequential:
            pass

class SimplePredictionModel:
    """
    A simple fallback prediction model when TensorFlow is not available.
    Uses exponential moving average for prediction.
    """
    def __init__(self, sequence_length=5):
        self.sequence_length = sequence_length
        self.is_trained = False
        self.alpha = 0.3  # Smoothing factor
        
    def train(self, simulation_data: Dict, epochs=None, batch_size=None):
        """Simple model doesn't need training, just stores the data"""
        self.simulation_data = simulation_data
        self.is_trained = True
        
    def predict_next_generations(self, simulation_data: Dict, num_predictions: int = 5) -> List[int]:
        """Predict using exponential smoothing"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
            
        neutron_counts = simulation_data["neutron_counts"]
        
        if len(neutron_counts) < 2:
            raise ValueError("Need at least 2 data points for prediction.")
        
        predictions = []
        last_values = neutron_counts[-self.sequence_length:]
        
        # Calculate growth rate from the last few values
        if len(last_values) >= 2:
            growth_factors = [last_values[i] / max(1, last_values[i-1]) for i in range(1, len(last_values))]
            avg_growth = sum(growth_factors) / len(growth_factors)
        else:
            avg_growth = 1.0
        
        # Use EMA with growth factor for prediction
        last_value = last_values[-1]
        
        for _ in range(num_predictions):
            next_val = int(max(0, round(last_value * avg_growth)))
            predictions.append(next_val)
            last_value = next_val
            
        return predictions
    
    def save_model(self, path=None):
        """Simple model doesn't need saving"""
        pass
        
    def load_model(self, path=None):
        """Simple model doesn't need loading"""
        pass

class NeutronPredictionModel:
    def __init__(self, sequence_length: int = 5):
        """
        Initialize the ML model for predicting neutron behavior.
        
        Args:
            sequence_length: Number of previous generations to consider for prediction
        """
        self.sequence_length = sequence_length
        self.is_trained = False
        
        # Use the appropriate model based on TensorFlow availability
        if TENSORFLOW_AVAILABLE:
            self.model = self._build_model()
            self.fallback = False
        else:
            self.simple_model = SimplePredictionModel(sequence_length)
            self.fallback = True
        
    def _build_model(self) -> Optional[Any]:
        """Build and compile the LSTM model."""
        if not TENSORFLOW_AVAILABLE:
            return None
            
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(self.sequence_length, 3), return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')  # Predicting neutron count
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def _prepare_sequences(self, data: List[int], fission_prob: float, neutrons_per_fission: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training/prediction.
        """
        if not TENSORFLOW_AVAILABLE:
            return None, None
            
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            # For each time step we include neutron count, fission probability and neutrons per fission
            seq = [[data[j], fission_prob, neutrons_per_fission] for j in range(i, i + self.sequence_length)]
            X.append(seq)
            y.append(data[i + self.sequence_length])
            
        return np.array(X), np.array(y)
    
    def train(self, simulation_data: Dict, epochs: int = 50, batch_size: int = 32):
        """
        Train the model using simulation data.
        """
        if self.fallback:
            self.simple_model.train(simulation_data)
            self.is_trained = True
            return
            
        neutron_counts = simulation_data["neutron_counts"]
        fission_prob = simulation_data["fission_probability"]
        neutrons_per_fission = simulation_data["neutrons_per_fission"]
        
        if len(neutron_counts) <= self.sequence_length:
            raise ValueError(f"Not enough data points. Need at least {self.sequence_length + 1}.")
        
        X, y = self._prepare_sequences(neutron_counts, fission_prob, neutrons_per_fission)
        
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        self.is_trained = True
    
    def predict_next_generations(self, simulation_data: Dict, num_predictions: int = 5) -> List[int]:
        """
        Predict neutron counts for future generations.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
            
        if self.fallback:
            return self.simple_model.predict_next_generations(simulation_data, num_predictions)
            
        neutron_counts = simulation_data["neutron_counts"]
        fission_prob = simulation_data["fission_probability"]
        neutrons_per_fission = simulation_data["neutrons_per_fission"]
        
        # Get the most recent sequence
        last_sequence = [[neutron_counts[i], fission_prob, neutrons_per_fission] 
                         for i in range(len(neutron_counts) - self.sequence_length, len(neutron_counts))]
        
        predictions = []
        current_sequence = np.array([last_sequence])
        
        for _ in range(num_predictions):
            next_val = int(max(0, round(self.model.predict(current_sequence, verbose=0)[0][0])))
            predictions.append(next_val)
            
            # Update sequence for the next prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1] = [next_val, fission_prob, neutrons_per_fission]
            
        return predictions
    
    def save_model(self, path: str = "neutron_model"):
        """Save the model to disk."""
        if self.fallback:
            return
            
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save(f"{path}/model.h5")
        
    def load_model(self, path: str = "neutron_model"):
        """Load the model from disk."""
        if self.fallback:
            return
            
        self.model = tf.keras.models.load_model(f"{path}/model.h5")
        self.is_trained = True
