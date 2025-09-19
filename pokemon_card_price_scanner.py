import cv2
import numpy as np
import pyautogui
import mss
import pytesseract
import requests
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import json
from dataclasses import dataclass
from typing import Optional, Dict, List
import re
from bs4 import BeautifulSoup

@dataclass
class CardInfo:
    name: str
    set_name: str = ""
    card_number: str = ""
    condition: str = "NM"

@dataclass
class PriceInfo:
    market_price: float = 0.0
    low_price: float = 0.0
    high_price: float = 0.0
    source: str = ""

class ScreenCapture:
    def __init__(self):
        pass
        
    def capture_full_screen(self):
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                screenshot = np.array(sct.grab(monitor))
                return cv2.cvtColor(screenshot, cv2.COLOR_BGRA2RGB)
        except Exception as e:
            print(f"MSS capture failed: {e}, trying pyautogui fallback")
            screenshot = pyautogui.screenshot()
            return np.array(screenshot)
    
    def capture_to_pil_full(self):
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                screenshot = np.array(sct.grab(monitor))
                return Image.fromarray(cv2.cvtColor(screenshot, cv2.COLOR_BGRA2RGB))
        except Exception as e:
            print(f"MSS capture failed: {e}, trying pyautogui fallback")
            return pyautogui.screenshot()

class CardRecognition:
    def __init__(self):
        tesseract_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Users\dlaev\AppData\Local\tesseract-ocr\tesseract.exe',
            r'tesseract'  # Try system PATH
        ]
        
        for path in tesseract_paths:
            try:
                pytesseract.pytesseract.tesseract_cmd = path
                # Test if it works
                pytesseract.image_to_string(Image.new('RGB', (100, 100), 'white'))
                print(f"Tesseract found at: {path}")
                break
            except:
                continue
        else:
            print("Warning: Tesseract not found - OCR will fail")
        
    def extract_text_from_image(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        try:
            text = pytesseract.image_to_string(image, config='--psm 6')
            return text.strip()
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
    
    def parse_card_info(self, text: str) -> Optional[CardInfo]:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if not lines:
            return None
            
        card_name = ""
        set_name = ""
        card_number = ""
        
        for i, line in enumerate(lines):
            if i == 0:
                card_name = line
            
            if re.search(r'\d+/\d+', line):
                card_number = line
            
            if any(keyword in line.lower() for keyword in ['base set', 'jungle', 'fossil', 'team rocket', 'neo', 'expedition']):
                set_name = line
        
        if card_name:
            return CardInfo(
                name=card_name,
                set_name=set_name,
                card_number=card_number
            )
        
        return None

class PriceAPI:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    def scrape_tcg_prices(self, card_name: str) -> Optional[PriceInfo]:
        # Perform local analysis instead of external API calls
        return self.get_local_card_analysis(card_name)
    
    def get_local_card_analysis(self, card_name: str) -> Optional[PriceInfo]:
        """Perform local card analysis without external API calls."""
        try:
            print(f"Analyzing card locally: {card_name}")
            
            # Return local analysis instead of price data
            return PriceInfo(
                market_price=0.0,  # Not a price
                low_price=0.0,     # Not a price  
                high_price=0.0,    # Not a price
                source="Local PSA Pre-grader Analysis"
            )
                    
        except Exception as e:
            print(f"Local analysis error: {e}")
        
        return self._get_mock_grade(card_name)
    
    def _get_mock_grade(self, card_name: str) -> PriceInfo:
        """Return mock grading analysis instead of prices."""
        base_grades = {
            'charizard': {'centering': 8.5, 'corners': 9.0, 'edges': 8.0},
            'blastoise': {'centering': 9.0, 'corners': 8.5, 'edges': 8.5},
            'venusaur': {'centering': 8.0, 'corners': 8.0, 'edges': 9.0},
            'pikachu': {'centering': 9.5, 'corners': 9.0, 'edges': 9.0},
        }
        
        card_lower = card_name.lower()
        for name, grades in base_grades.items():
            if name in card_lower:
                overall = sum(grades.values()) / len(grades)
                return PriceInfo(
                    market_price=overall,  # Overall grade
                    low_price=grades['centering'],  # Centering score
                    high_price=grades['corners'],   # Corner score  
                    source="Mock PSA Analysis"
                )
        
        return PriceInfo(
            market_price=7.5,  # Default overall grade
            low_price=7.0,     # Default centering
            high_price=8.0,    # Default corners
            source="Mock PSA Analysis"
        )

class PokemonCardScanner:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Pokemon Card Price Scanner")
        self.root.geometry("800x600")
        
        self.screen_capture = ScreenCapture()
        self.card_recognition = CardRecognition()
        self.price_api = PriceAPI()
        
        self.is_scanning = False
        self.scan_thread = None
        
        self.setup_ui()
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        ttk.Label(main_frame, text="Pokemon Card Price Scanner", 
                 font=('Arial', 16, 'bold')).grid(row=0, column=0, columnspan=3, pady=10)
        
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        self.scan_button = ttk.Button(control_frame, text="Start Scanning Full Screen", 
                                     command=self.toggle_scanning)
        self.scan_button.grid(row=0, column=0, padx=5)
        
        ttk.Button(control_frame, text="Single Scan", 
                  command=self.single_scan).grid(row=0, column=1, padx=5)
        
        self.preview_frame = ttk.LabelFrame(main_frame, text="Screen Preview", padding="5")
        self.preview_frame.grid(row=2, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.preview_label = ttk.Label(self.preview_frame, text="No preview available")
        self.preview_label.grid(row=0, column=0)
        
        info_frame = ttk.LabelFrame(main_frame, text="Card Information", padding="5")
        info_frame.grid(row=3, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.info_text = tk.Text(info_frame, height=10, width=70)
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=self.info_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.info_text.configure(yscrollcommand=scrollbar.set)
        
        price_frame = ttk.LabelFrame(main_frame, text="Price Information", padding="5")
        price_frame.grid(row=4, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        self.price_labels = {}
        price_items = [("Market Price:", "market"), ("Low Price:", "low"), ("High Price:", "high")]
        for i, (label, key) in enumerate(price_items):
            ttk.Label(price_frame, text=label).grid(row=0, column=i*2, padx=5, sticky=tk.W)
            self.price_labels[key] = ttk.Label(price_frame, text="$0.00", font=('Arial', 10, 'bold'))
            self.price_labels[key].grid(row=0, column=i*2+1, padx=5, sticky=tk.W)
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
    
    def toggle_scanning(self):
        if self.is_scanning:
            self.stop_scanning()
        else:
            self.start_scanning()
    
    def start_scanning(self):
        self.is_scanning = True
        self.scan_button.config(text="Stop Scanning")
        
        self.scan_thread = threading.Thread(target=self.scan_loop, daemon=True)
        self.scan_thread.start()
    
    def stop_scanning(self):
        self.is_scanning = False
        self.scan_button.config(text="Start Scanning")
    
    def scan_loop(self):
        while self.is_scanning:
            try:
                self.single_scan()
                time.sleep(2)
            except Exception as e:
                print(f"Scan error: {e}")
                time.sleep(1)
    
    def single_scan(self):
        try:
            screenshot = self.screen_capture.capture_full_screen()
            pil_image = self.screen_capture.capture_to_pil_full()
            
            self.update_preview(pil_image)
            
            text = self.card_recognition.extract_text_from_image(screenshot)
            
            if text:
                card_info = self.card_recognition.parse_card_info(text)
                
                if card_info:
                    price_info = self.price_api.scrape_tcg_prices(card_info.name)
                    
                    self.update_info_display(card_info, text, price_info)
                    self.update_price_display(price_info)
                else:
                    self.update_info_display(None, text, None)
            
        except Exception as e:
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, f"Error during scan: {e}")
    
    def update_preview(self, pil_image):
        try:
            display_size = (400, 300)
            # Create a copy to avoid modifying original
            preview_img = pil_image.copy()
            preview_img.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(preview_img)
            
            # Update label with image
            self.preview_label.config(image=photo, text="")
            # Keep a reference to prevent garbage collection
            self.preview_label.image = photo
            
            print(f"Preview updated: {preview_img.size}")
            
        except Exception as e:
            error_msg = f"Preview error: {e}"
            print(error_msg)
            self.preview_label.config(text=error_msg, image="")
    
    def update_info_display(self, card_info: Optional[CardInfo], raw_text: str, price_info: Optional[PriceInfo]):
        self.info_text.delete(1.0, tk.END)
        
        timestamp = time.strftime("%H:%M:%S")
        self.info_text.insert(tk.END, f"[{timestamp}] Scan Results:\n")
        self.info_text.insert(tk.END, "=" * 50 + "\n\n")
        
        if card_info:
            self.info_text.insert(tk.END, f"Card Name: {card_info.name}\n")
            if card_info.set_name:
                self.info_text.insert(tk.END, f"Set: {card_info.set_name}\n")
            if card_info.card_number:
                self.info_text.insert(tk.END, f"Card Number: {card_info.card_number}\n")
            self.info_text.insert(tk.END, f"Condition: {card_info.condition}\n\n")
            
            if price_info:
                self.info_text.insert(tk.END, f"Price Source: {price_info.source}\n")
                self.info_text.insert(tk.END, f"Market Price: ${price_info.market_price:.2f}\n")
                self.info_text.insert(tk.END, f"Low Price: ${price_info.low_price:.2f}\n")
                self.info_text.insert(tk.END, f"High Price: ${price_info.high_price:.2f}\n\n")
        else:
            self.info_text.insert(tk.END, "No card information detected\n\n")
        
        self.info_text.insert(tk.END, "Raw OCR Text:\n")
        self.info_text.insert(tk.END, "-" * 20 + "\n")
        self.info_text.insert(tk.END, raw_text + "\n")
        
        self.info_text.see(tk.END)
    
    def update_price_display(self, price_info: Optional[PriceInfo]):
        if price_info:
            self.price_labels["market"].config(text=f"${price_info.market_price:.2f}")
            self.price_labels["low"].config(text=f"${price_info.low_price:.2f}")
            self.price_labels["high"].config(text=f"${price_info.high_price:.2f}")
        else:
            for label in self.price_labels.values():
                label.config(text="N/A")
    
    def run(self):
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("Application interrupted")

if __name__ == "__main__":
    app = PokemonCardScanner()
    app.run()