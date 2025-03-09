"""
Helper script to install TensorFlow and run the nuclear simulation program.
This script automates both the TensorFlow installation and program execution.
"""

import os
import sys
import subprocess
import platform
import time

def main():
    """Install TensorFlow and then run the nuclear simulation program"""
    print("=" * 60)
    print("Nuclear Simulation with TensorFlow Setup")
    print("=" * 60)
    
    # First try to install TensorFlow
    print("\n[Step 1] Installing TensorFlow 2.18.0...")
    try:
        # Use the dedicated script for TensorFlow 2.18.0 installation
        if platform.system() == "Windows":
            tf_install_result = subprocess.run([sys.executable, "install_tf_2.18.0.py"], 
                                              capture_output=True, text=True)
        else:
            tf_install_result = subprocess.run([sys.executable, "install_tf_2.18.0.py"],
                                              capture_output=True, text=True)
        
        # Display the output
        print(tf_install_result.stdout)
        if tf_install_result.stderr:
            print("Errors encountered:", tf_install_result.stderr)
            print("\nTrying full TensorFlow installation with fallbacks...")
            
            # If the simple installer fails, try the more comprehensive one
            tf_install_result = subprocess.run([sys.executable, "install_tensorflow.py"],
                                               capture_output=True, text=True)
            print(tf_install_result.stdout)
    except Exception as e:
        print(f"Error during TensorFlow installation: {e}")
        print("Continuing with program execution using the fallback prediction model...")
    
    print("\n" + "="*60)
    print("[Step 2] Running Nuclear Simulation Program...")
    print("="*60 + "\n")
    
    # Determine the python command in virtual environment
    if platform.system() == "Windows":
        python_cmd = os.path.join("aienv", "Scripts", "python")
    else:
        python_cmd = os.path.join("aienv", "bin", "python")
    
    # Run the main program
    try:
        if os.path.exists(python_cmd + (".exe" if platform.system() == "Windows" else "")):
            # Run with virtual environment python
            subprocess.run([python_cmd, "main.py"])
        else:
            # Fallback to system python
            print("Virtual environment not found. Using system Python...")
            subprocess.run([sys.executable, "main.py"])
    except Exception as e:
        print(f"Error running the program: {e}")
    
    print("\nProgram execution completed.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    input("\nPress Enter to exit...")
