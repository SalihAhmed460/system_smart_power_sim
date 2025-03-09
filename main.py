import tkinter as tk
import sys
import os
import logging
import warnings
import argparse
import platform
from gui import NuclearSimulationApp

def setup_logging():
    """Set up logging configuration."""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, "nuclear_sim.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = ["numpy", "matplotlib", "psutil"]
    optional_packages = ["tensorflow", "scikit-learn"]
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_required.append(package)
    
    for package in optional_packages:
        try:
            __import__(package)
        except ImportError:
            missing_optional.append(package)
            
    if missing_required:
        print(f"Missing required packages: {', '.join(missing_required)}")
        print("Please install them using pip: pip install " + " ".join(missing_required))
        print("\nRun 'python install_dependencies.py' for automatic installation.")
        return False
    
    if missing_optional:
        print(f"Missing optional packages: {', '.join(missing_optional)}")
        print("For full functionality, please install: pip install " + " ".join(missing_optional))
        print("The program will run with reduced functionality.")
        
    return True

def handle_dependency_error():
    """Handle missing dependencies by offering to install them."""
    print("\nWould you like to install missing dependencies now? (y/n)")
    response = input().strip().lower()
    
    if response == 'y':
        print("Installing dependencies...")
        try:
            # Try to run the dependency installer
            if platform.system() == "Windows":
                os.system("python install_dependencies.py")
            else:
                os.system("python3 install_dependencies.py")
            print("Please restart the application.")
        except Exception as e:
            print(f"Error during installation: {e}")
    else:
        print("Please install the dependencies manually and restart the application.")

def main():
    """Main entry point for the application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Power Smart System - Nuclear Simulation")
    parser.add_argument('--dashboard', action='store_true', help='Launch the system dashboard')
    parser.add_argument('--install', action='store_true', help='Install dependencies before running')
    args = parser.parse_args()
    
    # Install dependencies if requested
    if args.install:
        try:
            if platform.system() == "Windows":
                os.system("python install_dependencies.py")
            else:
                os.system("python3 install_dependencies.py")
            return
        except Exception as e:
            print(f"Error during installation: {e}")
            return
    
    # Set up logging
    setup_logging()
    
    # Check dependencies
    if not check_dependencies():
        print("Critical dependencies are missing.")
        handle_dependency_error()
        input("Press Enter to exit...")
        return
        
    # Create the main window
    try:
        if args.dashboard:
            # Import and run the dashboard
            try:
                print("Starting Power Smart System Dashboard...")
                from dashboard import Dashboard, run_dashboard
                run_dashboard()
            except ImportError as e:
                print(f"Error loading dashboard module: {e}")
                print("Falling back to standard interface")
                # Fall back to standard interface
                root = tk.Tk()
                app = NuclearSimulationApp(root)
                root.update_idletasks()
                width = root.winfo_width()
                height = root.winfo_height()
                x = (root.winfo_screenwidth() // 2) - (width // 2)
                y = (root.winfo_screenheight() // 2) - (height // 2)
                root.geometry(f'+{x}+{y}')
                root.mainloop()
        else:
            # Run the standard GUI
            root = tk.Tk()
            app = NuclearSimulationApp(root)
            
            # Configure window sizing
            root.update_idletasks()
            width = root.winfo_width()
            height = root.winfo_height()
            x = (root.winfo_screenwidth() // 2) - (width // 2)
            y = (root.winfo_screenheight() // 2) - (height // 2)
            root.geometry(f'+{x}+{y}')
            
            # Run the application
            root.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
