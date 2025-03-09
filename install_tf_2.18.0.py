"""
Simple script to install TensorFlow 2.18.0 in the virtual environment
"""
import os
import sys
import subprocess
import platform

def main():
    # Determine the pip command based on platform
    if platform.system() == "Windows":
        pip_cmd = os.path.join("aienv", "Scripts", "pip")
        python_cmd = os.path.join("aienv", "Scripts", "python")
    else:
        pip_cmd = os.path.join("aienv", "bin", "pip")
        python_cmd = os.path.join("aienv", "bin", "python")
    
    # Check if virtual environment exists
    if not os.path.exists("aienv"):
        print("Creating virtual environment...")
        try:
            subprocess.check_call([sys.executable, "-m", "venv", "aienv"])
        except Exception as e:
            print(f"Failed to create virtual environment: {e}")
            return False
    
    print("Installing TensorFlow 2.18.0...")
    
    try:
        # First make sure we don't have conflicting installations
        try:
            subprocess.check_call([pip_cmd, "uninstall", "-y", "tensorflow", "tensorflow-cpu"])
        except:
            pass  # Ignore errors from uninstall
        
        # Install TensorFlow 2.18.0
        subprocess.check_call([pip_cmd, "install", "tensorflow==2.18.0"])
        
        # Verify installation
        verification_result = subprocess.run(
            [python_cmd, "-c", "import tensorflow as tf; print(f'TensorFlow {tf.__version__} successfully installed')"],
            capture_output=True,
            text=True
        )
        
        if verification_result.returncode == 0:
            print(verification_result.stdout.strip())
            print("\nTensorFlow 2.18.0 installation successful!")
            return True
        else:
            print(f"Installation verification failed: {verification_result.stderr}")
            return False
    
    except Exception as e:
        print(f"Error installing TensorFlow 2.18.0: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nYou can now run the nuclear simulation with full AI capabilities.")
        print("Run 'python run.py' to start the program.")
    else:
        print("\nTensorFlow installation failed. The program will run with reduced functionality.")
    
    input("Press Enter to exit...")
