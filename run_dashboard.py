"""
Launcher script for the Power Smart System Dashboard.
This script handles environment setup and launches the dashboard interface.
"""
import os
import sys
import subprocess
import platform
import time

def setup_environment():
    """Set up the runtime environment."""
    # Determine the activation script and python command based on platform
    if platform.system() == "Windows":
        python_cmd = os.path.join("aienv", "Scripts", "python") if os.path.exists("aienv") else sys.executable
        activate_cmd = os.path.join("aienv", "Scripts", "activate")
    else:
        python_cmd = os.path.join("aienv", "bin", "python") if os.path.exists("aienv") else sys.executable
        activate_cmd = os.path.join("aienv", "bin", "activate")
    
    print("=" * 60)
    print("Power Smart System Dashboard")
    print("=" * 60)
    
    # Check for virtual environment
    if os.path.exists("aienv"):
        print("Using existing virtual environment: aienv")
    else:
        print("Virtual environment not found. Using system Python.")
    
    return python_cmd

def main():
    """Main entry point."""
    python_cmd = setup_environment()
    
    # Run the dashboard
    print("\nLaunching Power Smart System Dashboard...")
    try:
        subprocess.run([python_cmd, "dashboard.py"])
    except Exception as e:
        print(f"Error launching dashboard: {e}")
        print("\nYou can try running it manually:")
        print("python dashboard.py")
    
    print("\nDashboard session ended.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    input("\nPress Enter to exit...")
