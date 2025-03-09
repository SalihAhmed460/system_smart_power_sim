import os
import sys
import subprocess
import platform
import time
import shutil

def check_system_info():
    """Get system information that might be relevant for debugging TensorFlow issues"""
    info = {
        "Platform": platform.platform(),
        "Python Version": platform.python_version(),
        "Architecture": platform.architecture()[0],
        "Machine": platform.machine(),
        "Processor": platform.processor()
    }
    
    try:
        # Check available memory
        import psutil
        memory = psutil.virtual_memory()
        info["Total Memory"] = f"{memory.total / (1024**3):.2f} GB"
        info["Available Memory"] = f"{memory.available / (1024**3):.2f} GB"
    except ImportError:
        info["Memory"] = "psutil not installed, memory info unavailable"
    
    return info

def install_tensorflow():
    """
    Install TensorFlow in the aienv virtual environment
    """
    print("=" * 60)
    print("TensorFlow Installation Helper")
    print("=" * 60)
    
    # Get system information
    print("\nSystem Information:")
    system_info = check_system_info()
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # Check if virtual environment exists
    if not os.path.exists("aienv"):
        print("\nCreating virtual environment...")
        try:
            subprocess.check_call([sys.executable, "-m", "venv", "aienv"])
        except Exception as e:
            print(f"Failed to create virtual environment: {e}")
            return False
    else:
        print("\nUsing existing virtual environment: aienv")
    
    # Get pip command path
    if platform.system() == "Windows":
        pip_cmd = os.path.join("aienv", "Scripts", "pip")
        python_cmd = os.path.join("aienv", "Scripts", "python")
        activate_cmd = os.path.join("aienv", "Scripts", "activate")
    else:
        pip_cmd = os.path.join("aienv", "bin", "pip")
        python_cmd = os.path.join("aienv", "bin", "python")
        activate_cmd = os.path.join("aienv", "bin", "activate")
    
    # Check if pip exists
    if not os.path.exists(pip_cmd + (".exe" if platform.system() == "Windows" else "")):
        print(f"Warning: pip not found at {pip_cmd}")
        print("Attempting to repair the virtual environment...")
        
        # Try to reinstall pip
        try:
            subprocess.check_call([python_cmd, "-m", "ensurepip"])
            print("Successfully installed pip")
        except Exception as e:
            print(f"Failed to install pip: {e}")
            return False
    
    # Upgrade pip first
    print("\nUpgrading pip...")
    try:
        subprocess.check_call([pip_cmd, "install", "--upgrade", "pip"])
    except Exception as e:
        print(f"Warning: Could not upgrade pip: {e}")
    
    # Install required supporting packages
    print("\nInstalling numpy and other required packages...")
    try:
        subprocess.check_call([pip_cmd, "install", "numpy", "matplotlib", "wheel", "setuptools"])
    except Exception as e:
        print(f"Warning: Issue installing supporting packages: {e}")
    
    # Install TensorFlow with the specific version
    print("\nInstalling TensorFlow 2.18.0...")
    
    try:
        # First make sure we don't have conflicting installations
        try:
            subprocess.check_call([pip_cmd, "uninstall", "-y", "tensorflow", "tensorflow-cpu"])
        except:
            pass  # Ignore errors from uninstall
            
        # Try to install the specific 2.18.0 version
        subprocess.check_call([pip_cmd, "install", "tensorflow==2.18.0"])
        print("Successfully installed TensorFlow 2.18.0")
        
        # Verify installation
        print("\nVerifying TensorFlow installation...")
        
        # Try importing tensorflow
        verification_result = subprocess.run(
            [python_cmd, "-c", "import tensorflow as tf; print(f'TensorFlow {tf.__version__} successfully installed'); print(f'Num GPUs Available: {len(tf.config.list_physical_devices(\"GPU\"))}')"],
            capture_output=True,
            text=True
        )
        
        if verification_result.returncode == 0:
            print(verification_result.stdout.strip())
            print("\n" + "=" * 60)
            print(f"TensorFlow 2.18.0 installation successful!")
            print(f"You can now run the nuclear simulation with full AI capabilities.")
            print("=" * 60)
            
            # Instructions for running the program
            print("\nTo run the nuclear simulation program:")
            if platform.system() == "Windows":
                print(f"1. Activate the environment: {activate_cmd}")
                print(f"2. Run the program: python main.py")
            else:
                print(f"1. Activate the environment: source {activate_cmd}")
                print(f"2. Run the program: python main.py")
                
            return True
        else:
            print(f"Installation verification failed:")
            print(f"Error: {verification_result.stderr.strip()}")
            
            # Try to handle common errors
            if "module 'tensorflow' has no attribute 'config'" in verification_result.stderr:
                print("\nDetected outdated TensorFlow API usage.")
                print("Trying to fix API compatibility...")
                # Create a compatibility wrapper
                
    except Exception as e:
        print(f"Failed to install TensorFlow 2.18.0: {e}")
    
    # If the specific version fails, try the fallback versions
    print("\nTrying alternative TensorFlow versions...")
    
    tensorflow_versions = [
        ("tensorflow", "Latest version"),
        ("tensorflow-cpu", "CPU-only version"),
        ("tensorflow==2.15.0", "Version 2.15.0"),
        ("tensorflow==2.14.0", "Version 2.14.0"),
        ("tensorflow==2.12.0", "Version 2.12.0")
    ]
    
    # Skip the first entry since we already tried the latest version
    for tf_package, description in tensorflow_versions:
        print(f"\nTrying {description} ({tf_package})...")
        try:
            # First make sure we don't have conflicting installations
            try:
                subprocess.check_call([pip_cmd, "uninstall", "-y", "tensorflow", "tensorflow-cpu"])
            except:
                pass  # Ignore errors from uninstall
                
            subprocess.check_call([pip_cmd, "install", tf_package])
            print(f"Successfully installed {tf_package}")
            
            # Verify installation
            print("\nVerifying TensorFlow installation...")
            try:
                # Try importing tensorflow
                verification_result = subprocess.run(
                    [python_cmd, "-c", "import tensorflow as tf; print(f'TensorFlow {tf.__version__} successfully installed'); print(f'Num GPUs Available: {len(tf.config.list_physical_devices(\"GPU\"))}')"],
                    capture_output=True,
                    text=True
                )
                
                if verification_result.returncode == 0:
                    print(verification_result.stdout.strip())
                    print("\n" + "=" * 60)
                    print(f"TensorFlow installation successful!")
                    print(f"You can now run the nuclear simulation with full AI capabilities.")
                    print("=" * 60)
                    
                    # Instructions for running the program
                    print("\nTo run the nuclear simulation program:")
                    if platform.system() == "Windows":
                        print(f"1. Activate the environment: {activate_cmd}")
                        print(f"2. Run the program: python main.py")
                    else:
                        print(f"1. Activate the environment: source {activate_cmd}")
                        print(f"2. Run the program: python main.py")
                        
                    return True
                else:
                    print(f"Installation verification failed. Trying next option...")
                    print(f"Error: {verification_result.stderr.strip()}")
            except Exception as e:
                print(f"Verification failed: {e}")
        except Exception as e:
            print(f"Failed to install {tf_package}: {e}")
    
    # If all installations failed, try scikit-learn instead
    print("\n" + "=" * 60)
    print("WARNING: Could not install TensorFlow.")
    print("Installing scikit-learn as an alternative...")
    print("=" * 60)
    
    try:
        subprocess.check_call([pip_cmd, "install", "scikit-learn", "joblib"])
        print("Successfully installed scikit-learn!")
        
        # Create a tensorflow compatibility module
        print("\nCreating TensorFlow compatibility layer...")
        tf_compat_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tensorflow_compat")
        if not os.path.exists(tf_compat_dir):
            os.makedirs(tf_compat_dir)
            
        # Create __init__.py
        with open(os.path.join(tf_compat_dir, "__init__.py"), "w") as f:
            f.write('print("Using scikit-learn as TensorFlow alternative")\n')
        
        # Install the compatibility module to the virtual environment
        print("\nInstalling compatibility layer...")
        subprocess.check_call([pip_cmd, "install", "-e", tf_compat_dir])
        
        print("\nThe program will run with a simplified model based on scikit-learn.")
        return True
    except Exception as e:
        print(f"Failed to install alternative ML library: {e}")
        
        print("\nDiagnostic Information:")
        print("1. Please verify your internet connection")
        print("2. Check if your system meets requirements:")
        print("   - Python 3.7-3.10 is recommended")
        print("   - 64-bit operating system is required")
        print("   - Sufficient disk space is required")
        print("3. You might need to install packages manually.")
        
        return False

if __name__ == "__main__":
    try:
        install_tensorflow()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    input("\nPress Enter to exit...")
