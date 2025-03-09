import os
import sys
import subprocess
import platform
import time

def create_virtual_env():
    """Create a virtual environment if it doesn't exist"""
    if not os.path.exists("aienv"):
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", "aienv"])
    else:
        print("Virtual environment already exists.")

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    # Determine the activation script and pip command based on platform
    if platform.system() == "Windows":
        pip_cmd = os.path.join("aienv", "Scripts", "pip")
        python_cmd = os.path.join("aienv", "Scripts", "python")
    else:
        pip_cmd = os.path.join("aienv", "bin", "pip")
        python_cmd = os.path.join("aienv", "bin", "python")
    
    # Upgrade pip first
    print("Upgrading pip...")
    try:
        if platform.system() == "Windows":
            subprocess.check_call([pip_cmd, "install", "--upgrade", "pip"])
        else:
            subprocess.check_call([pip_cmd, "install", "--upgrade", "pip"])
    except Exception as e:
        print(f"Warning: Could not upgrade pip: {e}")
    
    # Install packages with better error handling
    packages = ["numpy", "matplotlib"]
    
    # Try installing regular packages first
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([pip_cmd, "install", package])
            print(f"Successfully installed {package}")
        except Exception as e:
            print(f"Error installing {package}: {e}")
            return False
    
    # Special handling for TensorFlow - inform user but don't block program execution
    print("\n" + "-" * 60)
    print("NOTE: This program can use TensorFlow for advanced AI predictions.")
    print("TensorFlow installation is optional but recommended.")
    print("To install TensorFlow later, run: python install_tensorflow.py")
    print("-" * 60 + "\n")
    
    # Simple verification without TensorFlow
    print("Verifying required installations...")
    verification_script = "import numpy; import matplotlib; print('Required packages installed successfully.')"
    
    try:
        subprocess.check_call([python_cmd, "-c", verification_script])
        return True
    except Exception as e:
        print(f"Verification failed: {e}")
        return False

def run_program():
    """Run the nuclear simulation program"""
    print("Starting Nuclear Simulation Program...")
    
    # Run the program based on platform
    if platform.system() == "Windows":
        python_cmd = os.path.join("aienv", "Scripts", "python")
    else:
        python_cmd = os.path.join("aienv", "bin", "python")
    
    try:
        subprocess.check_call([python_cmd, "main.py"])
    except Exception as e:
        print(f"Error running the program: {e}")
        print("\nYou can try running the program directly with:")
        if platform.system() == "Windows":
            print("1. Activate the virtual environment: .\\aienv\\Scripts\\activate")
            print("2. Run the program: python main.py")
        else:
            print("1. Activate the virtual environment: source ./aienv/bin/activate")
            print("2. Run the program: python main.py")

if __name__ == "__main__":
    try:
        create_virtual_env()
        if install_requirements():
            print("Required dependencies installed successfully.")
            print("Starting program in 3 seconds...")
            time.sleep(3)  # Give user time to read the success message
            run_program()
        else:
            print("\nThere was an issue installing dependencies.")
            print("\nYou might need to install them manually:")
            print("1. Activate the virtual environment")
            if platform.system() == "Windows":
                print("   .\\aienv\\Scripts\\activate")
            else:
                print("   source ./aienv/bin/activate")
            print("2. Install the required packages:")
            print("   pip install numpy matplotlib")
            print("3. For AI functionality, also install:")
            print("   pip install tensorflow")
            print("\nAfterwards, run the program with: python main.py")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    input("\nPress Enter to exit...")
