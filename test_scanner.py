import tkinter as tk
from tkinter import ttk
import mss
import numpy as np
from PIL import Image, ImageTk

def test_capture():
    try:
        sct = mss.mss()
        monitor = sct.monitors[1]
        screenshot = np.array(sct.grab(monitor))
        print("Screen capture successful!")
        print(f"Screenshot shape: {screenshot.shape}")
        return True
    except Exception as e:
        print(f"Screen capture failed: {e}")
        return False

def create_gui():
    try:
        root = tk.Tk()
        root.title("Test Scanner")
        root.geometry("400x300")
        
        ttk.Label(root, text="Pokemon Card Scanner Test").pack(pady=20)
        ttk.Button(root, text="Test Capture", command=test_capture).pack(pady=10)
        ttk.Button(root, text="Close", command=root.quit).pack(pady=10)
        
        print("GUI created successfully!")
        root.mainloop()
        
    except Exception as e:
        print(f"GUI creation failed: {e}")

if __name__ == "__main__":
    print("Starting test...")
    create_gui()
    print("Test complete.")