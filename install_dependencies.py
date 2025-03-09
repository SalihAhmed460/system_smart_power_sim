"""
Dependency installer for Power Smart System.
This script installs all required packages for the simulation system.
"""
import os
import sys
import subprocess
import platform
import time

def check_and_create_venv():
    """Check if virtual environment exists and create if needed."""
    print("Checking virtual environment...")
    
    # Create venv if it doesn't exist
    if not os.path.exists("aienv"):
        print("Creating virtual environment 'aienv'...")
        try:
            subprocess.check_call([sys.executable, "-m", "venv", "aienv"])
            print("Virtual environment created successfully!")
        except Exception as e:
            print(f"Failed to create virtual environment: {e}")
            print("Will install packages in system Python instead.")
            return False
        
    return True

def get_pip_cmd():
    """Get the pip command based on platform and environment."""
    if os.path.exists("aienv"):
        # Use virtual environment pip
        if platform.system() == "Windows":
            return os.path.join("aienv", "Scripts", "pip")
        else:
            return os.path.join("aienv", "bin", "pip")
    else:
        # Use system pip
        return "pip"

def get_python_cmd():
    """Get the python command based on platform and environment."""
    if os.path.exists("aienv"):
        # Use virtual environment python
        if platform.system() == "Windows":
            return os.path.join("aienv", "Scripts", "python")
        else:
            return os.path.join("aienv", "bin", "python")
    else:
        # Use system python
        return sys.executable

def install_required_packages():
    """Install all required packages."""
    pip_cmd = get_pip_cmd()
    
    print("\n=== Installing Required Packages ===")
    required = ["numpy", "matplotlib", "psutil"]
    
    for package in required:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([pip_cmd, "install", package])
            print(f"✅ {package} installed successfully!")
        except Exception as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    return True

def install_optional_packages():
    """Install optional packages like TensorFlow."""
    pip_cmd = get_pip_cmd()
    
    print("\n=== Installing Optional Packages ===")
    print("Attempting to install TensorFlow...")
    
    try:
        # First try the default tensorflow package
        subprocess.check_call([pip_cmd, "install", "tensorflow"])
        print("✅ TensorFlow installed successfully!")
        return True
    except Exception as e:
        print(f"Could not install default TensorFlow: {e}")
        
        # Try alternatives
        alternatives = [
            "tensorflow-cpu",
            "tensorflow==2.15.0",
            "tensorflow==2.12.0"
        ]
        
        for alt_package in alternatives:
            try:
                print(f"Trying alternative: {alt_package}")
                subprocess.check_call([pip_cmd, "install", alt_package])
                print(f"✅ {alt_package} installed successfully!")
                return True
            except Exception as e:
                print(f"Failed to install {alt_package}: {e}")
                
        # If all tensorflow options fail, install scikit-learn as fallback
        print("Installing scikit-learn as TensorFlow alternative...")
        try:
            subprocess.check_call([pip_cmd, "install", "scikit-learn", "joblib"])
            print("✅ scikit-learn installed as TensorFlow alternative")
        except Exception as e:
            print(f"Failed to install scikit-learn: {e}")
        
        return False

def verify_installation():
    """Verify package installation."""
    python_cmd = get_python_cmd()
    
    print("\n=== Verifying Installation ===")
    
    # Check required packages
    try:
        cmd = "import numpy; import matplotlib; print('Required packages verified!')"
        result = subprocess.run([python_cmd, "-c", cmd], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All required packages are installed correctly!")
            return True
        else:
            print(f"❌ Verification failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False

def main():
    """Main function to install all dependencies."""
    print("=" * 60)
    print("Power Smart System Dependency Installer")
    print("=" * 60)
    
    # Create virtual environment
    using_venv = check_and_create_venv()
    
    # Install required packages
    if not install_required_packages():
        print("\nSome required packages could not be installed.")
        print("Please try to install them manually:")
        print("pip install numpy matplotlib psutil")
        return False
    
    # Install optional packages
    install_optional_packages()  # Continue even if this fails
    
    # Verify installation
    if verify_installation():
        print("\n" + "=" * 60)
        print("✅ Installation complete! You can now run the program.")
        
        if using_venv:
            if platform.system() == "Windows":
                print("\nTo activate the virtual environment:")
                print("   .\\aienv\\Scripts\\activate")
            else:
                print("\nTo activate the virtual environment:")
                print("   source ./aienv/bin/activate")
        
        print("\nTo run the dashboard:")
        print("   python main.py --dashboard")
        print("\nOr use the standard interface:")
        print("   python main.py")
        print("=" * 60)
        return True
    else:
        print("\nVerification failed. Some packages may not be installed correctly.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        
        if success:
            # Offer to run the program
            response = input("\nWould you like to run the dashboard now? (y/n): ").strip().lower()
            if response == 'y':
                python_cmd = get_python_cmd()
                print("\nStarting Power Smart System Dashboard...")
                subprocess.call([python_cmd, "main.py", "--dashboard"])
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    input("\nPress Enter to exit...")
