"""
Theme and styling module for the Power Smart System application.
Provides consistent UI styling across the application.
"""
import tkinter as tk
from tkinter import ttk
import os
import platform

# Color palette
COLORS = {
    "primary": "#1976d2",       # Blue
    "secondary": "#388e3c",     # Green
    "warning": "#f57c00",       # Orange
    "error": "#d32f2f",         # Red
    "success": "#388e3c",       # Green
    "info": "#0288d1",          # Light Blue
    "dark": "#212121",          # Almost Black
    "light_bg": "#f5f5f5",      # Light Gray
    "dark_bg": "#263238",       # Dark Blue-Gray
    "text_light": "#ffffff",    # White
    "text_dark": "#212121",     # Almost Black
    "border": "#bdbdbd",        # Medium Gray
    "hover": "#e1f5fe",         # Very Light Blue
}

# Images directory
def get_assets_dir():
    """Get the absolute path to the assets directory."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# Style configuration
def apply_theme(root):
    """Apply the custom theme to the application."""
    style = ttk.Style()
    
    # Try to load a modern theme available on the platform
    available_themes = style.theme_names()
    preferred_themes = ['clam', 'alt', 'vista', 'xpnative', 'winnative', 'default']
    
    # Use the first preferred theme that's available
    selected_theme = 'clam'  # Default fallback
    for theme in preferred_themes:
        if theme in available_themes:
            selected_theme = theme
            break
    
    style.theme_use(selected_theme)
    
    # Configure common elements
    style.configure("TFrame", background=COLORS["light_bg"])
    style.configure("TLabel", background=COLORS["light_bg"], foreground=COLORS["text_dark"])
    style.configure("TLabelframe", background=COLORS["light_bg"], foreground=COLORS["primary"])
    style.configure("TLabelframe.Label", background=COLORS["light_bg"], foreground=COLORS["primary"], font=("Arial", 10, "bold"))
    
    # Configure buttons
    style.configure("TButton", 
                   background=COLORS["primary"], 
                   foreground=COLORS["text_light"],
                   borderwidth=0,
                   focusthickness=3,
                   focuscolor=COLORS["primary"])
    
    # Button states
    style.map("TButton",
             background=[("active", COLORS["info"]), ("disabled", COLORS["border"])],
             foreground=[("disabled", COLORS["text_dark"])])
    
    # Progress bar
    style.configure("TProgressbar", 
                   background=COLORS["primary"],
                   troughcolor=COLORS["light_bg"],
                   borderwidth=0,
                   thickness=10)
    
    # Special button styles
    style.configure("Success.TButton", background=COLORS["success"])
    style.map("Success.TButton", background=[("active", COLORS["secondary"])])
    
    style.configure("Warning.TButton", background=COLORS["warning"])
    style.map("Warning.TButton", background=[("active", COLORS["warning"])])
    
    # Entry fields
    style.configure("TEntry", fieldbackground=COLORS["light_bg"], borderwidth=1)
    
    # Tree view
    style.configure("Treeview", 
                   background=COLORS["light_bg"],
                   fieldbackground=COLORS["light_bg"],
                   foreground=COLORS["text_dark"])
    
    style.configure("Treeview.Heading", 
                   background=COLORS["primary"],
                   foreground=COLORS["text_light"],
                   font=("Arial", 10, "bold"))
    
    # Configure root window
    if platform.system() == 'Windows':
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)  # Make the application DPI aware
        except ImportError:
            pass
    
    root.configure(bg=COLORS["light_bg"])
    
    # Return style and colors for further customization
    return style, COLORS

def get_font(size=10, bold=False, italic=False):
    """Get a standard font with the specified properties."""
    font_style = ""
    if bold:
        font_style += " bold"
    if italic:
        font_style += " italic"
    
    return ("Arial", size, font_style.strip())

# Create logo and application icons
def create_logo(master, width=100, height=100):
    """Create a logo canvas for the application."""
    logo_canvas = tk.Canvas(master, width=width, height=height, 
                         bg=COLORS["light_bg"], highlightthickness=0)
    
    # Outer circle
    logo_canvas.create_oval(10, 10, width-10, height-10, 
                         outline=COLORS["primary"], width=4)
    
    # Inner circle
    logo_canvas.create_oval(30, 30, width-30, height-30, 
                         outline=COLORS["secondary"], width=3)
    
    # Center dot
    logo_canvas.create_oval(width/2-5, height/2-5, width/2+5, height/2+5, 
                         fill=COLORS["primary"], outline="")
    
    # Electron orbits
    logo_canvas.create_oval(20, 20, width-20, height-20, 
                         outline=COLORS["info"], width=1)
    
    # Electrons
    logo_canvas.create_oval(width/2-40, height/2, width/2-30, height/2+10, 
                         fill=COLORS["warning"], outline="")
    logo_canvas.create_oval(width/2+30, height/2-10, width/+40, height/2, 
                         fill=COLORS["warning"], outline="")
    logo_canvas.create_oval(width/2, height/2-40, width/2+10, height/2-30, 
                         fill=COLORS["warning"], outline="")
    
    return logo_canvas
