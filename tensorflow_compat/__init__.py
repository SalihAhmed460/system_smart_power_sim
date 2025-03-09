"""
TensorFlow compatibility layer using scikit-learn.
This module provides a simplified interface to simulate TensorFlow functionality
for systems where TensorFlow can't be installed.
"""
import warnings
import numpy as np
import joblib
import os

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    warnings.warn("scikit-learn could not be imported. Using minimal fallback model.")
    SKLEARN_AVAILABLE = False

print("Using scikit-learn as TensorFlow alternative")

# Define a minimal class for TensorFlow version compatibility
class __version__:
    __version__ = "compat.1.0.0"

__version__ = "compat.1.0.0"

class Sequential:
    """A simplified Sequential model class that uses scikit-learn instead of TensorFlow."""
    
    def __init__(self, layers=None):
        """Initialize the model."""
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_compiled = False
        self.is_trained = False
        
    def add(self, layer):
        """Add a layer (ignored, for compatibility only)."""
        pass
        
    def compile(self, optimizer=None, loss=None, metrics=None):
        """Compile the model (just sets a flag, for compatibility)."""
        if SKLEARN_AVAILABLE:
            self.model = LinearRegression()
        self.is_compiled = True
        
    def fit(self, x, y, epochs=1, batch_size=None, verbose=0):
        """Train the model using scikit-learn's LinearRegression."""
        if not SKLEARN_AVAILABLE:
            warnings.warn("scikit-learn not available, training skipped.")
            return
            
        # Reshape input if needed
        x_reshaped = x.reshape(x.shape[0], -1)  # Flatten any multi-dimensional features
        
        # Scale the data
        x_scaled = self.scaler.fit_transform(x_reshaped)
        
        # Train the model
        self.model.fit(x_scaled, y)
        self.is_trained = True
        
    def predict(self, x, verbose=0):
        """Make predictions using the trained model."""
        if not SKLEARN_AVAILABLE or not self.is_trained:
            # Return a simple average prediction if no model
            return np.mean(x, axis=1, keepdims=True)
            
        # Reshape input if needed
        x_reshaped = x.reshape(x.shape[0], -1)  # Flatten any multi-dimensional features
        
        # Scale the data
        x_scaled = self.scaler.transform(x_reshaped)
        
        # Make predictions
        return self.model.predict(x_scaled).reshape(-1, 1)
        
    def save(self, filepath):
        """Save the model to a file."""
        if not SKLEARN_AVAILABLE or not self.is_trained:
            return
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model and scaler
        joblib.dump({'model': self.model, 'scaler': self.scaler}, filepath)
        
    def load_weights(self, filepath):
        """Load model weights from a file."""
        if not SKLEARN_AVAILABLE:
            return
            
        if os.path.exists(filepath):
            saved_data = joblib.load(filepath)
            self.model = saved_data['model']
            self.scaler = saved_data['scaler']
            self.is_trained = True

# Create a dummy keras module
class keras:
    class models:
        Sequential = Sequential
        
    class layers:
        class Dense:
            def __init__(self, units, activation=None, input_shape=None):
                self.units = units
                self.activation = activation
                
        class LSTM:
            def __init__(self, units, activation=None, input_shape=None, return_sequences=False):
                self.units = units
                self.activation = activation
                self.return_sequences = return_sequences
                
        class Dropout:
            def __init__(self, rate):
                self.rate = rate
                
    class optimizers:
        class Adam:
            def __init__(self, learning_rate=0.001):
                self.learning_rate = learning_rate

# Create dummy functions to simulate TensorFlow functionality
def config():
    class _DeviceConfig:
        def list_physical_devices(self, device_type):
            return []
    return _DeviceConfig()

# Make keras accessible as tf.keras
keras = keras
