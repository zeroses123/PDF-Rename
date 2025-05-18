#!/usr/bin/env python3
import os
import sys
import subprocess
import glob
import re
import shutil
import logging
import argparse
from typing import Dict, List, Optional, Tuple, Any, Set
import json
import platform
from datetime import datetime
import yaml
import tkinter as tk
from tkinter import messagebox
import socket
import time

# Now import the installed packages
try:
    import pytesseract
    import pdf2image
    import PyPDF2
    import requests
    import numpy as np
    from PIL import Image, ImageChops
    from skimage.metrics import structural_similarity as ssim
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please make sure all required libraries are installed.")
    sys.exit(1)

# Path setup for EXE bundle
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Check if we're running as an executable or as a Python script
if hasattr(sys, 'frozen'):
    # Running as executable
    base_dir = APP_DIR
else:
    # Running as script
    base_dir = os.getcwd()

# Find poppler and tesseract
LOCAL_TESS = os.path.join(APP_DIR, "tesseract", "tesseract.exe")
LOCAL_TESSDATA = os.path.join(APP_DIR, "tesseract", "tessdata")

# Default Poppler location
POPPLER_BIN = os.path.join(APP_DIR, "resources", "poppler", "bin")

# Check if resources are in the current working directory instead
if not os.path.exists(POPPLER_BIN):
    alt_poppler_bin = os.path.join(base_dir, "resources", "poppler", "bin")
    if os.path.exists(alt_poppler_bin):
        POPPLER_BIN = alt_poppler_bin
        
if os.path.exists(LOCAL_TESS):
    pytesseract.pytesseract.tesseract_cmd = LOCAL_TESS
    os.environ.setdefault("TESSDATA_PREFIX", LOCAL_TESSDATA)

# ======================================================================
# CONFIGURATION
# ======================================================================

# Default configuration if config.yaml is not found
DEFAULT_CONFIG = {
    "global": {
        "openai_key": "youropenaiapikey",
        "model": "gpt-4.1-mini",
        "threshold": 0.5,
        "aggressive_split": False,
        "processing_mode": "BUNDLE",
        "blank_page_threshold": 0.90,
        "directories": {
            "complete": "Rechnungen",
            "incomplete": "Rechnungen-Unsicher",
            "backup": "Backup",
            "temp": "temp_invoices"
        },
        "error_handling": {
            "show_popup": True,
            "keep_files_on_error": True
        }
    },
    "company_names": [
        "Vattenfall Europe Sales GmbH",
        "VHV Versicherung",
        "VHV Allgemeine Versicherung AG",
        "Telekom Deutschland GmbH",
        "Deutsche Telekom AG",
        "SECURITAS Alert Services GmbH",
        "DB Fernverkehr AG",
        "Deutsche Lufthansa AG",
        "reuter europe GmbH"
    ],
    "similarity_weights": {
        "header_similarity": 0.2,
        "footer_similarity": 0.1,
        "margin_similarity": 0.1,
        "logo_similarity": 0.3,
        "text_similarity": 0.3,
        "page_number_match": 0.2,
        "company_name_match": 0.4
    },
    "folders": []
}

# Global configuration variable
CONFIG = {}

def load_config():
    """Load configuration from config.yaml or create default if not exists"""
    global CONFIG
    
    config_path = os.path.join(os.getcwd(), "config.yaml")
    
    # Check if config file exists
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                CONFIG = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            logger.info("Using default configuration")
            CONFIG = DEFAULT_CONFIG
    else:
        # Create default config file
        logger.info(f"Configuration file not found. Creating default at {config_path}")
        CONFIG = DEFAULT_CONFIG
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(CONFIG, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            logger.error(f"Error creating default configuration file: {e}")

# Load similarity weights and other settings from config
SIM_WEIGHTS = {}
BLANK_PAGE_THRESHOLD = 0.95
AGGRESSIVE_SPLIT = False

def show_error_popup(title, message):
    """Display an error popup if enabled in configuration"""
    if CONFIG.get("global", {}).get("error_handling", {}).get("show_popup", True):
        try:
            # Create a root window but hide it
            root = tk.Tk()
            root.withdraw()
            
            # Show the error message
            messagebox.showerror(title, message)
            
            # Destroy the root window
            root.destroy()
        except Exception as e:
            logger.error(f"Failed to show error popup: {e}")

def check_internet_connection():
    """Check if there is an internet connection available"""
    try:
        # Try to connect to OpenAI's API
        socket.create_connection(("api.openai.com", 443), timeout=5)
        return True
    except OSError:
        return False

# Get company names from configuration
def get_company_names():
    """Get the list of company names from configuration"""
    return CONFIG.get("company_names", [])

# Get similarity threshold from configuration
def get_similarity_threshold():
    """Get the similarity threshold from configuration"""
    return CONFIG.get("global", {}).get("threshold", 0.5)

# Get processing mode from configuration
def get_processing_mode():
    """Get the processing mode from configuration"""
    return CONFIG.get("global", {}).get("processing_mode", "BUNDLE")

# Get blank page threshold from configuration
def get_blank_page_threshold():
    """Get the blank page detection threshold from configuration"""
    return CONFIG.get("global", {}).get("blank_page_threshold", 0.95)

# Get similarity weights from configuration
def get_similarity_weights():
    """Get the similarity weights from configuration"""
    return CONFIG.get("similarity_weights", {
        "header_similarity": 0.2,
        "footer_similarity": 0.1,
        "margin_similarity": 0.1,
        "logo_similarity": 0.3,
        "text_similarity": 0.3,
        "page_number_match": 0.2,
        "company_name_match": 0.4
    })

# Get directory paths from configuration
def get_directory_path(directory_type):
    """Get the path for a specific directory type from configuration"""
    directories = CONFIG.get("global", {}).get("directories", {})
    default_dirs = {
        "complete": "Rechnungen",
        "incomplete": "Rechnungen-Unsicher",
        "backup": "Backup",
        "temp": "temp_invoices"
    }
    
    return directories.get(directory_type, default_dirs.get(directory_type, ""))

# Function to check and install required packages
def install_required_packages():
    required_packages = ['pytesseract', 'pdf2image', 'PyPDF2', 'requests', 'pillow', 'scikit-image', 'numpy', 'openai']
    installed_packages = []
    
    try:
        # Get list of installed packages
        reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
        installed_packages = [r.decode().split('==')[0].lower() for r in reqs.split()]
    except Exception as e:
        print(f"Warning: Could not check installed packages: {e}")
    
    # Install missing packages
    for package in required_packages:
        if package.lower() not in installed_packages:
            try:
                logger.info(f"Installing missing package: {package}")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            except Exception as e:
                logger.error(f"Failed to install {package}: {e}")
    
    logger.info("Required packages check completed.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('invoice_extractor.log')
    ]
)
logger = logging.getLogger(__name__)

# Load configuration first
load_config()

# Initialize global variables from config
SIM_WEIGHTS = CONFIG.get("similarity_weights", DEFAULT_CONFIG["similarity_weights"])
BLANK_PAGE_THRESHOLD = CONFIG["global"].get("blank_page_threshold", 0.90)
AGGRESSIVE_SPLIT = CONFIG["global"].get("aggressive_split", False)
os.environ.setdefault("AGGRESSIVE_INVOICE_SPLIT", str(AGGRESSIVE_SPLIT).lower())

# Install required packages
install_required_packages()

# On macOS, use system binaries if available for Poppler
if platform.system() == "Darwin":  # macOS
    # Check if poppler is installed via homebrew
    if os.path.exists("/opt/homebrew/bin/pdftoppm"):
        POPPLER_BIN = "/opt/homebrew/bin"
        logger.info(f"Using macOS Homebrew Poppler binaries from: {POPPLER_BIN}")
    elif os.path.exists("/usr/local/bin/pdftoppm"):
        POPPLER_BIN = "/usr/local/bin"
        logger.info(f"Using macOS Poppler binaries from: {POPPLER_BIN}")

# Set tesseract path based on operating system
def setup_tesseract():
    """Configure Tesseract OCR based on the operating system"""
    system = platform.system()
    
    # Log paths for debugging
    logger.info(f"POPPLER_BIN path: {POPPLER_BIN}")
    logger.info(f"POPPLER_BIN exists: {os.path.exists(POPPLER_BIN)}")
    if os.path.exists(POPPLER_BIN):
        try:
            poppler_contents = os.listdir(POPPLER_BIN)
            logger.info(f"Contents of POPPLER_BIN: {poppler_contents}")
            # Check for pdftoppm executable
            if platform.system() == "Darwin":
                if "pdftoppm" in poppler_contents:
                    logger.info(f"Found pdftoppm in POPPLER_BIN")
                else:
                    logger.warning(f"pdftoppm not found in POPPLER_BIN. PDF processing may fail.")
            else:
                if "pdftoppm.exe" in poppler_contents:
                    logger.info(f"Found pdftoppm.exe in POPPLER_BIN")
                else:
                    logger.warning(f"pdftoppm.exe not found in POPPLER_BIN. PDF processing may fail.")
        except Exception as e:
            logger.warning(f"Could not list Poppler directory contents: {e}")
    
    if system == "Windows":
        # Common installation paths on Windows
        windows_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Tesseract-OCR\tesseract.exe',
            # Add the Tesseract path if installed via Windows package managers
            os.path.join(os.environ.get('LOCALAPPDATA', ''), r'Programs\Tesseract-OCR\tesseract.exe'),
            os.path.join(os.environ.get('PROGRAMFILES', ''), r'Tesseract-OCR\tesseract.exe'),
            os.path.join(os.environ.get('PROGRAMFILES(X86)', ''), r'Tesseract-OCR\tesseract.exe')
        ]
        
        for path in windows_paths:
            if os.path.exists(path):
                logger.info(f"Found Tesseract at: {path}")
                pytesseract.pytesseract.tesseract_cmd = path
                return True
                
        # Check if tesseract is in PATH
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract found in system PATH")
            return True
        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract OCR not found on Windows system")
            logger.error("Please install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
            logger.error("Install to the default location: C:\\Program Files\\Tesseract-OCR\\")
            logger.error("Or add the Tesseract installation directory to your system PATH")
            logger.error("After installation, restart this script")
            
            # Create a more user-friendly error message for Windows users
            print("\n" + "="*60)
            print("ERROR: Tesseract OCR not found on your Windows system")
            print("="*60)
            print("\nTo fix this:")
            print("1. Download and install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
            print("2. Make sure to CHECK the 'Add to PATH' option during installation")
            print("3. Install to the default location: C:\\Program Files\\Tesseract-OCR\\")
            print("4. Restart your computer after installation")
            print("5. Run this script again\n")
            return False
            
    elif system == "Darwin":  # macOS
        # Try to set the path to tesseract on macOS with Homebrew
        mac_paths = [
            '/opt/homebrew/bin/tesseract',
            '/usr/local/bin/tesseract'
        ]
        
        for path in mac_paths:
            if os.path.exists(path):
                logger.info(f"Found Tesseract at: {path}")
                pytesseract.pytesseract.tesseract_cmd = path
                return True
                
        # Check if tesseract is in PATH
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract found in system PATH")
            return True
        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract OCR not found on macOS")
            logger.error("Install it with: brew install tesseract")
            return False
            
    else:  # Linux and other systems
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract found in system PATH")
            return True
        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract OCR not found on your system")
            logger.error("On Linux, install with: sudo apt-get install tesseract-ocr")
            return False

# Try to setup Tesseract
if not setup_tesseract():
    print("Failed to configure Tesseract OCR. Exiting.")
    sys.exit(1)
def extract_text_from_image_array(image: Image.Image) -> str:
    """Extract text from a PIL Image object using OCR."""
    try:
        text = pytesseract.image_to_string(image, lang='deu')
        return text
    except Exception as e:
        logger.error(f"Error extracting text from image: {e}")
        return ""

def extract_text_from_image(image_path: str) -> str:
    """Extract text from an image file using OCR."""
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang='deu')
        return text
    except Exception as e:
        logger.error(f"Error extracting text from image {image_path}: {e}")
        return ""

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using OCR."""
    try:
        # First try to extract text directly (if the PDF has text layers)
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text() or ""
        
        # If we got meaningful text, return it
        if text.strip() and len(text) > 100:  # Arbitrary threshold to determine if text extraction worked
            return text
        
        # If direct extraction didn't yield good results, use OCR
        logger.info(f"Using OCR for PDF: {pdf_path}")
        all_text = ""
        
        # Add debugging information about Poppler path
        if not os.path.exists(POPPLER_BIN):
            logger.warning(f"Poppler directory not found at: {POPPLER_BIN}")
        else:
            logger.info(f"Using Poppler from: {POPPLER_BIN}")
            # Remove poppler scripts output
            # try:
            #     contents = os.listdir(POPPLER_BIN)
            #     logger.info(f"Contents of Poppler directory: {contents}")
            # except Exception as e:
            #     logger.warning(f"Could not list Poppler directory contents: {e}")
        
        # Convert PDF to images
        images = pdf2image.convert_from_path(pdf_path, poppler_path=POPPLER_BIN)
        
        # Process each page
        for i, image in enumerate(images):
            # Extract text directly from the image object without saving temporarily
            page_text = extract_text_from_image_array(image)
            all_text += f"\n\n--- Page {i+1} ---\n\n" + page_text
        
        return all_text
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""
def query_llm(text: str) -> Dict[str, Any]:
    """
    Query OpenAI to extract invoice information.
    """
    # Check internet connectivity before making API call
    if not check_internet_connection():
        error_msg = "No internet connection available. Cannot process invoice."
        logger.error(error_msg)
        show_error_popup("Connection Error", error_msg)
        return {"error": error_msg, "no_internet": True}
    
    # Get company names from configuration
    company_names = get_company_names()
    company_names_str = ", ".join(company_names)
    
    # Prepare the prompt for OpenAI
    prompt = f"""
    Extract the following information from this invoice text:
    - Invoice date (Rechnungsdatum)
    - Invoice number (Rechnungsnummer)
    - Company name (Firmenname)
    - Invoice amount (Rechnungsbetrag)
    

    Return only the extracted information in JSON format:
    {{
        "invoice_date": "extracted date",
        "invoice_number": "extracted number",
        "company_name": "extracted company name",
        "invoice_amount": "extracted amount"
    }}
    Important Information:
    The company name will never be Bodenkontor Liegenschaften GmbH or similar. Use the other company you have found in the invoice.
    
    Here are the most likely company names that should appear on the invoice:
    {company_names_str}
    
    If you find one of these names in the invoice, please use it. If not, extract the most likely company name from the invoice.
    
    Here is the invoice text:
    {text}
    """
    
    # Call the OpenAI API
    return query_openai(prompt)

def query_openai(prompt: str) -> Dict[str, Any]:
    """Use OpenAI API to extract invoice information from invoice text."""
    try:
        # Get OpenAI API key from config, environment, or hardcoded key
        openai_api_key = CONFIG.get("global", {}).get("openai_key", "")
        if not openai_api_key:
            openai_api_key = os.environ.get('OPENAI_API_KEY', OPENAI_API_KEY)

        if not openai_api_key:
            error_msg = "No OpenAI API key found in config, environment, or script"
            logger.error(error_msg)
            show_error_popup("API Key Error", error_msg)
            return {"error": error_msg}
            
        # Basic validation of API key format
        if not (openai_api_key.startswith('sk-') and len(openai_api_key) > 20):
            error_msg = "Invalid OpenAI API key format. API keys should start with 'sk-' and be at least 20 characters."
            logger.error(error_msg)
            show_error_popup("API Key Error", error_msg)
            return {"error": error_msg}
        # Import the openai module
        try:
            import openai
            # Set the API key
            openai.api_key = openai_api_key
            client = openai.OpenAI(api_key=openai_api_key)
        except ImportError:
            return {"error": "Failed to import openai module. Please make sure it's installed with 'pip install openai'."}

        # Get model from config
        model = CONFIG.get("global", {}).get("model", "gpt-4.1-mini")
        logger.info(f"Sending invoice text to OpenAI API ({model})...")
        
        try:
            response = client.chat.completions.create(
                model=model,  # Use the model from config
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts information from invoices."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1  # Low temperature for more deterministic results
            )
            
            # Extract content from OpenAI response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                llm_response = response.choices[0].message.content
                
                # Add detailed logging of the raw response
                logger.info(f"Raw OpenAI response: {llm_response}")
                
                # Try to parse JSON from the response
                try:
                    # Find the start and end of JSON in the response
                    start_idx = llm_response.find('{')
                    end_idx = llm_response.rfind('}') + 1
                    
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = llm_response[start_idx:end_idx]
                        logger.info(f"Extracted JSON string: {json_str}")
                        try:
                            result = json.loads(json_str)
                            return result
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON parsing error: {e}")
                            return {"error": f"Failed to parse JSON from OpenAI response: {e}", "raw_response": llm_response}
                    else:
                        logger.error(f"No JSON found in response: {llm_response}")
                        return {"error": "Could not find JSON in OpenAI response", "raw_response": llm_response}
                except Exception as e:
                    logger.error(f"Error processing OpenAI response: {e}")
                    return {"error": f"Error processing OpenAI response: {e}", "raw_response": llm_response}
            else:
                logger.error(f"Unexpected response format: {response}")
                return {"error": "Unexpected response format from OpenAI", "raw_response": str(response)}
                
        except openai.BadRequestError as e:
            return {"error": f"OpenAI API Bad Request Error: {str(e)}"}
        except openai.AuthenticationError as e:
            return {"error": f"OpenAI API Authentication Error: {str(e)}"}
        except openai.RateLimitError as e:
            return {"error": f"OpenAI API Rate Limit Error: {str(e)}"}
        except openai.APIError as e:
            return {"error": f"OpenAI API Error: {str(e)}"}
    
    except Exception as e:
        logger.error(f"Unexpected error setting up OpenAI request: {str(e)}")
        return {"error": f"Unexpected error setting up OpenAI request: {str(e)}"}

def is_blank_page(image, threshold=None, debug=False):
    """
    Determine if an image is blank/white using multiple detection methods.
    
    Args:
        image: PIL Image object
        threshold: Brightness threshold (0-1), higher values mean stricter blank detection
                  If None, uses the value from config
        debug: If True, saves the image with debug information
    
    Returns:
        bool: True if the page is blank, False otherwise
    """
    # Use global threshold from config if none specified
    if threshold is None:
        threshold = BLANK_PAGE_THRESHOLD
        
    # Convert to grayscale
    gray_img = image.convert('L')
    
    # Calculate the average brightness (0-255)
    hist = gray_img.histogram()
    total_pixels = sum(hist)
    if total_pixels == 0:
        logger.debug("Empty image (no pixels)")
        return True
        
    # Calculate weighted brightness average
    brightness_sum = sum(i * pixel_count for i, pixel_count in enumerate(hist))
    average_brightness = brightness_sum / total_pixels
    
    # Normalize to 0-1 range
    normalized_brightness = average_brightness / 255.0
    
    # Check if the image has very low contrast (another indicator of blank page)
    dark_pixels = sum(hist[:50])  # Count very dark pixels
    dark_pixel_percentage = dark_pixels / total_pixels
    bright_pixels = sum(hist[200:])  # Count very bright pixels
    bright_pixel_percentage = bright_pixels / total_pixels
    low_contrast = bright_pixel_percentage > 0.95  # Page is >95% bright pixels
    
    # Calculate the standard deviation of the image to detect variation
    # (blank pages have low standard deviation)
    pixel_values = np.array(gray_img)
    std_dev = np.std(pixel_values) / 255.0
    low_variation = std_dev < 0.05
    
    # Initial assessment based on brightness and contrast
    initial_blank = normalized_brightness > threshold and low_contrast and low_variation
    
    logger.debug(f"Page analysis: brightness={normalized_brightness:.2f}, "
                f"contrast_bright={bright_pixel_percentage:.2f}, "
                f"contrast_dark={dark_pixel_percentage:.2f}, "
                f"std_dev={std_dev:.4f}")
    
    # If the initial test suggests it might be blank, perform deeper analysis
    if initial_blank:
        # Divide the image into a 3x3 grid and check each cell for content
        width, height = image.size
        cell_width = width // 3
        cell_height = height // 3
        
        # Check each cell for content
        has_content_in_cells = False
        for x in range(3):
            for y in range(3):
                cell = gray_img.crop((x*cell_width, y*cell_height, 
                                     (x+1)*cell_width, (y+1)*cell_height))
                cell_pixels = np.array(cell)
                cell_std_dev = np.std(cell_pixels) / 255.0
                
                # If any cell has significant variation, the page has content
                if cell_std_dev > 0.08:
                    has_content_in_cells = True
                    logger.debug(f"Found content in grid cell ({x},{y}): std_dev={cell_std_dev:.4f}")
                    break
            if has_content_in_cells:
                break
        
        # Final decision
        is_blank = initial_blank and not has_content_in_cells
        
        # For borderline cases, run quick OCR on a small sample
        if is_blank and (normalized_brightness < threshold + 0.05 or std_dev > 0.03):
            try:
                # Run OCR on a scaled-down version for speed
                small_img = image.resize((image.width // 2, image.height // 2))
                text = pytesseract.image_to_string(small_img, config='--psm 6')
                # If we found meaningful text, it's not blank
                if len(text.strip()) > 10:  # Arbitrary threshold for "meaningful" text
                    logger.debug(f"OCR found text on seemingly blank page: {text[:50]}...")
                    is_blank = False
            except Exception as e:
                logger.warning(f"OCR check failed: {e}")
        
        if debug and is_blank:
            # Save the "blank" image for manual inspection
            debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug")
            os.makedirs(debug_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            debug_path = os.path.join(debug_dir, f"blank_page_{timestamp}.png")
            image.save(debug_path)
            logger.debug(f"Saved blank page for inspection at: {debug_path}")
    else:
        is_blank = False
    
    if is_blank:
        logger.info(f"Page classified as blank: brightness={normalized_brightness:.2f}, "
                   f"contrast={bright_pixel_percentage:.2f}, std_dev={std_dev:.4f}")
    
    return is_blank

def compare_logo_area(img1, img2):
    """
    Compare the top-right corner (logo area) of two images and return a similarity score (0-1).
    Higher score means more similar logo areas.
    """
    # Resize images to same dimensions if they're different
    if img1.size != img2.size:
        img2 = img2.resize(img1.size)
    
    # Extract the logo area (top-right portion of the image - larger area)
    h, w = img1.height, img1.width
    # Using a larger area: 60% width, 25% height (instead of 25% width, 15% height)
    crop1 = img1.crop((w*0.6, 0, w, h*0.25))
    crop2 = img2.crop((w*0.6, 0, w, h*0.25))
    
    return compare_images(crop1, crop2)

def compare_images(img1, img2):
    """
    Compare two images and return a similarity score (0-1).
    Higher score means more similar images.
    """
    # Resize images to same dimensions if they're different
    if img1.size != img2.size:
        img2 = img2.resize(img1.size)
    
    # Convert to grayscale for comparison
    img1_gray = img1.convert('L')
    img2_gray = img2.convert('L')
    
    # Convert to numpy arrays
    img1_array = np.array(img1_gray)
    img2_array = np.array(img2_gray)
    
    # Calculate structural similarity index
    try:
        score, _ = ssim(img1_array, img2_array, full=True)
        return score
    except Exception as e:
        logger.error(f"Error comparing images: {e}")
        # Fall back to a simpler difference method
        diff = ImageChops.difference(img1_gray, img2_gray)
        diff_stats = ImageChops.difference(img1_gray, img2_gray).getbbox()
        if diff_stats is None:  # Images are identical
            return 1.0
        else:
            # Calculate a simple difference score (lower is more similar)
            hist = diff.histogram()
            sq = (value * ((idx % 256) ** 2) for idx, value in enumerate(hist))
            sum_sq = sum(sq)
            rms = (sum_sq / float(img1.size[0] * img1.size[1])) ** 0.5
            # Convert to similarity score (1 - normalized difference)
            return 1 - min(rms / 128, 1.0)

def compare_text_content(text1: str, text2: str) -> float:
    """
    Compare two text contents and return a similarity score (0-1).
    Higher score means more similar texts.
    """
    # Preprocess text to remove whitespace and convert to lowercase
    text1 = ' '.join(text1.lower().split())
    text2 = ' '.join(text2.lower().split())
    
    # If either text is empty, return 0
    if not text1 or not text2:
        return 0.0
    
    # Split into words
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    # Calculate Jaccard similarity (intersection over union)
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    if union == 0:
        return 0.0
    
    return intersection / union

def detect_page_numbers(text: str) -> Optional[Tuple[int, int]]:
    """
    Detect page numbering patterns like "Seite 1/2" or "Page 1 of 2".
    Returns a tuple of (current_page, total_pages) if found, None otherwise.
    """
    # German patterns
    patterns = [
        r'Seite\s+(\d+)\s*/\s*(\d+)',           # Seite 1/2
        r'Seite\s+(\d+)\s+von\s+(\d+)',         # Seite 1 von 2
        r'Blatt\s+(\d+)\s*/\s*(\d+)',           # Blatt 1/2
        
        # English patterns
        r'Page\s+(\d+)\s*/\s*(\d+)',            # Page 1/2
        r'Page\s+(\d+)\s+of\s+(\d+)',           # Page 1 of 2
        
        # Generic patterns
        r'(\d+)\s*/\s*(\d+)\s+Seite',           # 1/2 Seite
        r'(\d+)\s*/\s*(\d+)\s+Page',            # 1/2 Page
        r'(\d+)\s*-\s*(\d+)'                    # 1-2 (assuming first is current, second is total)
    ]
    
    for pattern in patterns:
        matches = re.search(pattern, text, re.IGNORECASE)
        if matches:
            try:
                current_page = int(matches.group(1))
                total_pages = int(matches.group(2))
                return (current_page, total_pages)
            except (ValueError, IndexError):
                continue
    
    return None

def extract_possible_company_names(text: str) -> Set[str]:
    """
    Attempt to extract possible company names from text.
    Returns a set of potential company names found.
    """
    company_names = set()
    
    # Look for common company type indicators (German and international)
    company_types = [
        # German company types
        'GmbH', 'AG', 'KG', 'OHG', 'e.V.', 'e. V.', 'GbR', 'UG', 'SE', 
        'Co. KG', 'Co.KG', 'mbH', 'haftungsbeschränkt', 'eG',
        # International company types
        'Ltd', 'Inc', 'LLC', 'B.V.', 'Corp', 'Corporation', 'S.A.', 'S.p.A.',
        'N.V.', 'GesmbH', 'Ges.m.b.H', 'S.A.S.', 'S.r.l.'
    ]
    
    # Known company names from config
    known_companies = CONFIG.get("company_names", [])
    
    # Check for known companies first (most reliable)
    for company in known_companies:
        if company.lower() in text.lower():
            company_names.add(company)
            logger.debug(f"Found known company name: {company}")
    
    # Regular expression to find company names (with company type suffix)
    for company_type in company_types:
        pattern = fr'([A-Z][a-zA-Z0-9\s\.\-&]+)\s+{re.escape(company_type)}'
        matches = re.finditer(pattern, text)
        for match in matches:
            company_name = match.group(0).strip()
            if 5 < len(company_name) < 50:  # Reasonable length for a company name
                company_names.add(company_name)
                logger.debug(f"Found company name with pattern '{company_type}': {company_name}")
    
    # Check for "typical" letterhead patterns (at the beginning of document)
    lines = text.split('\n')
    
    # Analyze first few lines (usually contain letterhead)
    for i, line in enumerate(lines[:10]):
        line = line.strip()
        if not line:
            continue
            
        # Letterhead detection: First line is often a company name if it has uppercase letters
        if i <= 3 and line and line[0].isupper() and 5 < len(line) < 50:
            # Check if contains any company type
            for company_type in company_types:
                if company_type in line:
                    company_names.add(line)
                    logger.debug(f"Found company name in letterhead: {line}")
                    break
            
            # Check for "standard" letterhead (company name on first line)
            if i == 0 and not any(c_type in line for c_type in company_types):
                # Assume a standalone first line could be a company name
                words = len(line.split())
                if 2 <= words <= 6 and not line.endswith(':') and not line.startswith('Re:'):
                    company_names.add(line)
                    logger.debug(f"Potential company name from first line: {line}")
    
    # Check for bill-to/ship-to sections
    bill_to_patterns = [
        r'(?:bill\s*to|rechnungsadresse)[:\s]+([A-Z][\w\s\.,&-]+)',
        r'(?:customer|kunde)[:\s]+([A-Z][\w\s\.,&-]+)'
    ]
    
    for pattern in bill_to_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            if match.group(1):
                name_part = match.group(1).strip()
                # Limit to the first line or up to a comma/period if multi-line
                first_line = name_part.split('\n')[0].split(',')[0].split('.')[0].strip()
                if 5 < len(first_line) < 50:
                    company_names.add(first_line)
                    logger.debug(f"Found company name in bill-to section: {first_line}")

    # If we didn't find any company name, try a more aggressive approach with common words
    if not company_names:
        # Words commonly found in company names
        company_indicators = [
            'Service', 'Systems', 'Solutions', 'Technologies', 'Consulting',
            'Versicherung', 'Versicherungen', 'Gruppe', 'Group', 'Bank',
            'Telekom', 'Media', 'Logistik', 'Energie', 'Wasser', 'Stadtwerke'
        ]
        
        for indicator in company_indicators:
            pattern = fr'([A-Z][a-zA-Z0-9\s\.\-&]+\s+{indicator})'
            matches = re.finditer(pattern, text)
            for match in matches:
                company_name = match.group(0).strip()
                if 5 < len(company_name) < 50:
                    company_names.add(company_name)
                    logger.debug(f"Found company name with indicator '{indicator}': {company_name}")
    
    # Log extracted company names for debugging
    if company_names:
        logger.debug(f"Extracted company names: {', '.join(company_names)}")
    else:
        logger.debug("No company names extracted from text")
        
    return company_names

def split_pdf_into_invoices(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Split a multi-page PDF into individual invoice documents.
    Returns a list of dictionaries containing text and image data for each invoice.
    """
    logger.info(f"Analyzing multi-page PDF: {pdf_path}")
    
    try:
        # Add debugging information about Poppler path
        if not os.path.exists(POPPLER_BIN):
            logger.warning(f"Poppler directory not found at: {POPPLER_BIN}")
        else:
            logger.info(f"Using Poppler from: {POPPLER_BIN}")
            # Debug info about system
            logger.info(f"Platform: {platform.system()}")
            # Check if necessary Poppler binaries exist
            if platform.system() == "Darwin":  # macOS
                pdftoppm_path = os.path.join(POPPLER_BIN, "pdftoppm")
                if os.path.exists(pdftoppm_path):
                    logger.info(f"Found pdftoppm at: {pdftoppm_path}")
                else:
                    logger.warning(f"pdftoppm not found at: {pdftoppm_path}")
            else:  # Windows
                pdftoppm_path = os.path.join(POPPLER_BIN, "pdftoppm.exe")
                if os.path.exists(pdftoppm_path):
                    logger.info(f"Found pdftoppm.exe at: {pdftoppm_path}")
                else:
                    logger.warning(f"pdftoppm.exe not found at: {pdftoppm_path}")
                
        # Convert PDF to images
        images = pdf2image.convert_from_path(pdf_path, poppler_path=POPPLER_BIN)
        if not images:
            logger.error(f"Failed to convert PDF to images: {pdf_path}")
            return [{"error": "Failed to convert PDF to images", "text": ""}]
        
        # Use aggressive splitting from config or environment variable
        aggressive_split = AGGRESSIVE_SPLIT or os.environ.get('AGGRESSIVE_INVOICE_SPLIT', 'false').lower() == 'true'
        
        # Get threshold from config or environment variable
        invoice_similarity_threshold = float(os.environ.get('INVOICE_SIMILARITY_THRESHOLD', '0.5'))
        
        logger.info(f"Using aggressive splitting: {aggressive_split}, similarity threshold: {invoice_similarity_threshold}")
        
        # If aggressive splitting is enabled, treat each page as a separate invoice
        if aggressive_split:
            invoices = []
            for i, image in enumerate(images):
                # Skip blank pages
                if is_blank_page(image, debug=True):
                    logger.info(f"Page {i+1}: Blank page - skipping")
                    continue
                    
                page_text = extract_text_from_image_array(image)
                logger.info(f"Page {i+1}: Treating as separate invoice (aggressive splitting)")
                invoices.append({
                    "images": [image],
                    "text": page_text
                })
            
            logger.info(f"Split PDF into {len(invoices)} invoices using aggressive splitting")
            return invoices
        
        # Otherwise, use the more sophisticated approach
        # Initialize variables
        invoices = []
        
        # Check if all pages are blank
        all_blank = True
        for img in images:
            if not is_blank_page(img, debug=True):
                all_blank = False
                break
                
        if all_blank:
            logger.warning("All pages appear to be blank. Processing as a single invoice.")
            invoices.append({
                "images": images,
                "text": "All pages appear to be blank."
            })
            return invoices
                
        # Start with the first non-blank page
        first_page_idx = 0
        for idx, img in enumerate(images):
            if not is_blank_page(img, debug=True):
                first_page_idx = idx
                break
                
        current_invoice_images = [images[first_page_idx]]
        current_invoice_text = extract_text_from_image_array(images[first_page_idx])
        possible_company_names = extract_possible_company_names(current_invoice_text)
        
        logger.info(f"Starting with page {first_page_idx + 1} (skipped {first_page_idx} blank pages)")
        
        # Process each page after the first non-blank page
        for i in range(first_page_idx + 1, len(images)):
            page_image = images[i]
            
            # Check if this is a blank page - if so, skip it
            if is_blank_page(page_image, debug=True):
                logger.info(f"Page {i+1}: Detected as blank/white page - ignoring")
                continue
                
            page_text = extract_text_from_image_array(page_image)
            
            # Default to assuming this is a new invoice unless proven otherwise
            is_new_invoice = True
            confidence_scores = []
            
            # Check header similarity (top 15%)
            header_region1 = current_invoice_images[-1].crop((0, 0, current_invoice_images[-1].width, int(current_invoice_images[-1].height * 0.15)))
            header_region2 = page_image.crop((0, 0, page_image.width, int(page_image.height * 0.15)))
            header_similarity = compare_images(header_region1, header_region2)
            confidence_scores.append(('header_similarity', header_similarity))
            
            # Check footer similarity (bottom 10%)
            footer_region1 = current_invoice_images[-1].crop((0, int(current_invoice_images[-1].height * 0.9), 
                                                          current_invoice_images[-1].width, current_invoice_images[-1].height))
            footer_region2 = page_image.crop((0, int(page_image.height * 0.9), 
                                          page_image.width, page_image.height))
            footer_similarity = compare_images(footer_region1, footer_region2)
            confidence_scores.append(('footer_similarity', footer_similarity))
            
            # Check left margin (first 10% width)
            margin_region1 = current_invoice_images[-1].crop((0, 0, int(current_invoice_images[-1].width * 0.1), current_invoice_images[-1].height))
            margin_region2 = page_image.crop((0, 0, int(page_image.width * 0.1), page_image.height))
            margin_similarity = compare_images(margin_region1, margin_region2)
            confidence_scores.append(('margin_similarity', margin_similarity))
            
            # Compare logo area in top-right corner of the page
            logo_similarity = compare_logo_area(current_invoice_images[-1], page_image)
            confidence_scores.append(('logo_similarity', logo_similarity))
            logger.info(f"Page {i+1}: Logo similarity score: {logo_similarity:.2f}")
            
            # Compare text content
            text_similarity = compare_text_content(current_invoice_text, page_text)
            confidence_scores.append(('text_similarity', text_similarity))
            
            # Check for page numbers
            page_numbers = detect_page_numbers(page_text)
            prev_page_numbers = detect_page_numbers(current_invoice_text)
            
            if page_numbers and prev_page_numbers:
                # If sequential page numbers and same total, likely same invoice
                if (page_numbers[1] == prev_page_numbers[1] and 
                    page_numbers[0] == prev_page_numbers[0] + 1):
                    page_number_match = 1.0  # High confidence
                else:
                    page_number_match = 0.0  # Low confidence
            else:
                page_number_match = 0.5  # Neutral confidence
            
            confidence_scores.append(('page_number_match', page_number_match))
            
            # Check for company name continuation
            page_company_names = extract_possible_company_names(page_text)
            company_name_match = 0.0
            
            logger.info(f"Page {i+1}: Extracted company names: {list(page_company_names)}")
            logger.info(f"Previous page company names: {list(possible_company_names)}")
            
            # Method 1: Direct company name matching
            if possible_company_names and page_company_names:
                common_names = possible_company_names.intersection(page_company_names)
                if common_names:
                    company_name_match = 0.9  # High confidence
                    logger.info(f"Page {i+1}: Exact company name match found: {next(iter(common_names))}")
                else:
                    logger.info(f"Page {i+1}: No exact company name match found")
                    
                    # Method 2: If no exact match, try fuzzy company name matching
                    # This handles slight OCR errors or formatting differences
                    best_similarity = 0.0
                    best_match_pair = None
                    
                    for prev_name in possible_company_names:
                        for curr_name in page_company_names:
                            # Calculate string similarity using token sort ratio (word order independent)
                            # Convert to lowercase for better matching
                            prev_lower = prev_name.lower()
                            curr_lower = curr_name.lower()
                            
                            # Simple word overlap metric
                            prev_words = set(prev_lower.split())
                            curr_words = set(curr_lower.split())
                            
                            if prev_words and curr_words:  # Avoid division by zero
                                overlap = len(prev_words.intersection(curr_words))
                                similarity = overlap / max(len(prev_words), len(curr_words))
                                
                                if similarity > best_similarity:
                                    best_similarity = similarity
                                    best_match_pair = (prev_name, curr_name)
                    
                    # If significant word overlap found
                    if best_similarity >= 0.5:  # At least 50% word overlap
                        company_name_match = 0.7  # Medium-high confidence
                        logger.info(f"Page {i+1}: Fuzzy company match found: {best_match_pair[0]} ≈ {best_match_pair[1]} (similarity: {best_similarity:.2f})")
            
            # Method 3: Fallback - check if company names appear in each other's text
            # This helps when OCR quality varies between pages
            if company_name_match == 0.0 and possible_company_names:
                for name in possible_company_names:
                    # Check if a company name from the previous page appears in the current page text
                    if name.lower() in page_text.lower():
                        company_name_match = 0.6  # Medium confidence
                        logger.info(f"Page {i+1}: Company name '{name}' from previous page found in current page text")
                        break
            
            # Method 4: Final fallback - if text similarity is very high, assume same company
            if company_name_match == 0.0 and text_similarity > 0.7:
                company_name_match = 0.5  # Medium confidence based on high text similarity
                logger.info(f"Page {i+1}: Assuming company match based on high text similarity ({text_similarity:.2f})")
            
            confidence_scores.append(('company_name_match', company_name_match))
            
            # Compute overall confidence based on all factors using weights from config
            overall_confidence = 0.0
            
            # Use SIM_WEIGHTS from config instead of hardcoded weights
            total_weight = 0.0
            
            # Create detailed output of similarity values for finetuning
            detailed_scores = "Page comparison values for finetuning:\n"
            for factor, score in confidence_scores:
                weight = SIM_WEIGHTS.get(factor, 0.1)  # Default weight if not specified
                detailed_scores += f"  {factor}: {score:.3f} (weight: {weight:.2f})\n"
                
                if factor in SIM_WEIGHTS:
                    overall_confidence += score * weight
                    total_weight += weight
            
            if total_weight > 0:
                overall_confidence /= total_weight
            
            logger.info(detailed_scores)
            logger.info(f"Page {i+1}: Overall confidence = {overall_confidence:.3f}, threshold = {invoice_similarity_threshold:.3f}")
            
            # Decision: is this a continuation or a new invoice?
            if overall_confidence > invoice_similarity_threshold:
                is_new_invoice = False
                logger.info(f"Page {i+1}: Treating as continuation of current invoice (confidence: {overall_confidence:.3f})")
            else:
                logger.info(f"Page {i+1}: Treating as new invoice (confidence below threshold: {overall_confidence:.3f})")
            
            # Process based on whether this is a new invoice or continuation
            if is_new_invoice:
                # Save the current invoice and start a new one
                invoices.append({
                    "images": current_invoice_images,
                    "text": current_invoice_text
                })
                
                # Start new invoice
                current_invoice_images = [page_image]
                current_invoice_text = page_text
                possible_company_names = extract_possible_company_names(page_text)
                logger.info(f"Page {i+1}: Detected as start of new invoice")
            else:
                # Add to current invoice
                current_invoice_images.append(page_image)
                current_invoice_text += f"\n\n--- Page {i+1} ---\n\n" + page_text
                # Update company names with any new ones found
                possible_company_names.update(extract_possible_company_names(page_text))
        
        # Add the final invoice
        invoices.append({
            "images": current_invoice_images,
            "text": current_invoice_text
        })
        
        return invoices
        
    except Exception as e:
        logger.error(f"Error splitting PDF into invoices: {e}")
        
        # Special handling for Windows poppler errors
        if platform.system() == "Windows" and "poppler" in str(e).lower():
            logger.error("Extract the downloaded ZIP file and add the 'bin' directory to your PATH environment variable")
            
            print("\n" + "="*60)
            print("ERROR: PDF processing requires poppler utilities on Windows")
            print("="*60)
            print("\nTo fix this:")
            print("1. Download poppler for Windows from: https://github.com/oschwartz10612/poppler-windows/releases/")
            print("2. Extract the ZIP file to a location like C:\\poppler")
            print("3. Add the bin directory (e.g., C:\\poppler\\bin) to your PATH environment variable")
            print("4. Restart your computer")
            print("5. Run this script again\n")
        
        return [{"error": f"Error processing PDF: {str(e)}", "text": ""}]

def sanitize_filename(filename: str) -> str:
    """
    Sanitize a string to be used as a filename.
    """
    # Handle None values
    if filename is None:
        return "Unknown"
        
    # Convert to string if not already
    if not isinstance(filename, str):
        filename = str(filename)
    
    # Replace invalid characters
    invalid_chars = r'<>:"/\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Trim long filenames
    if len(filename) > 200:
        filename = filename[:200]
    
    return filename.strip()
def generate_invoice_filename(data: Dict[str, Any], original_filename: str) -> Tuple[str, bool]:
    """
    Generate a filename based on invoice data.
    Returns a tuple of (filename, is_complete)
    """
    # Get values, using placeholders for missing data
    company = data.get('company_name', None) or 'Unknown'
    invoice_num = data.get('invoice_number', None) or 'NoNum'
    amount = data.get('invoice_amount', None) or 'NoAmount'
    
    # Log problematic values for debugging
    if data.get('invoice_amount') is None:
        logger.warning(f"Invoice amount is None in data: {data}")
    
    # Process date
    date_str = data.get('invoice_date', '')
    date_formatted = "XXXX-XX-XX"  # Default format if parsing fails
    
    if date_str:
        # Try to parse and standardize the date format
        try:
            # Handle common German date formats (DD.MM.YYYY)
            if re.match(r'\d{1,2}\.\d{1,2}\.\d{2,4}', date_str):
                parts = date_str.split('.')
                if len(parts) == 3:
                    day = parts[0].zfill(2)
                    month = parts[1].zfill(2)
                    year = parts[2]
                    if len(year) == 2:
                        year = '20' + year  # Assume 20xx for two-digit years
                    date_formatted = f"{year}-{month}-{day}"
            
            # Handle ISO format (YYYY-MM-DD)
            elif re.match(r'\d{4}-\d{1,2}-\d{1,2}', date_str):
                parts = date_str.split('-')
                if len(parts) == 3:
                    year = parts[0]
                    month = parts[1].zfill(2)
                    day = parts[2].zfill(2)
                    date_formatted = f"{year}-{month}-{day}"
            
            # Handle other formats (e.g., "6 Feb 2025" or "February 6, 2025")
            else:
                try:
                    # Try parsing with datetime
                    for fmt in ['%d %b %Y', '%B %d, %Y', '%d.%m.%Y', '%Y-%m-%d']:
                        try:
                            dt = datetime.strptime(date_str, fmt)
                            date_formatted = dt.strftime('%Y-%m-%d')
                            break
                        except ValueError:
                            continue
                except:
                    # Keep the original if all parsing attempts fail
                    date_formatted = date_str
        except Exception as e:
            logger.warning(f"Error parsing date '{date_str}': {e}")
            date_formatted = date_str
    
    # Create the filename
    filename = f"{sanitize_filename(company)}-{sanitize_filename(invoice_num)}-{sanitize_filename(amount)}-{sanitize_filename(date_formatted)}.pdf"
    
    # Check if we have all the required information
    is_complete = all([
        company != 'Unknown',
        invoice_num != 'NoNum',
        amount != 'NoAmount',
        date_formatted != 'XXXX-XX-XX' and date_formatted != date_str
    ])
    
    return (filename, is_complete)

def process_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Process a single file and extract invoice information.
    Returns a list of results, one for each invoice found in the file.
    """
    logger.info(f"\nProcessing: {file_path}")
    results = []
    
    # Determine file type and extract text/process accordingly
    file_ext = os.path.splitext(file_path)[1].lower()
    results = []  # Initialize results list here to ensure it's always defined
    
    try:
        # Add special debugging for the problematic file
        if os.path.basename(file_path) == "074_Rechnung_2025-03-03_100160234138_V87824159.pdf":
            logger.info(f"Processing problematic file: {file_path}")
            
        if file_ext in ['.pdf']:
            # Check if this might be a multi-invoice PDF
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                page_count = len(reader.pages)
            
            logger.info(f"PDF has {page_count} pages")
            
            # If PROCESSING_MODE is set to 'PDF_AS_INVOICE', treat the entire PDF as one invoice
            if PROCESSING_MODE == 'PDF_AS_INVOICE':
                logger.info(f"Processing entire PDF as a single invoice (PDF_AS_INVOICE mode)")
                
                # Extract text from all pages
                all_text = extract_text_from_pdf(file_path)
                
                # Query LLM for invoice information
                logger.info(f"Extracted {len(all_text)} characters of text. Querying LLM...")
                result = query_llm(all_text)
                
                # Add file info to the result
                result["original_file"] = file_path
                result["page_count"] = page_count
                
                results.append(result)
                return results
            
            # For multi-page PDFs in BUNDLE mode, try to split into individual invoices
            elif page_count > 1:
                logger.info(f"Attempting to split multi-page PDF: {file_path}")
                invoices = split_pdf_into_invoices(file_path)
                
                # Process each detected invoice
                for i, invoice in enumerate(invoices):
                    invoice_text = invoice["text"]
                    if not invoice_text.strip():
                        logger.warning(f"No text extracted from invoice {i+1} in {file_path}")
                        continue
                    
                    # Query LLM for invoice information
                    # Query OpenAI for invoice information
                    logger.info(f"Extracted {len(invoice_text)} characters from invoice {i+1}. Sending to OpenAI...")
                    result = query_llm(invoice_text)
                    
                    # Add detailed logging of the result
                    logger.info(f"OpenAI result: {result}")
                    
                    # Add additional info to the result
                    result["original_file"] = file_path
                    result["page_count"] = len(invoice["images"])
                    # Ensure result is a dictionary
                    if not isinstance(result, dict):
                        logger.warning(f"OpenAI returned unexpected result type: {type(result)}")
                        result = {"error": "Invalid response format from OpenAI", "original_file": file_path}
                        
                    # Save the extracted invoice as a new PDF if it contains multiple pages
                    # Save the extracted invoice as a new PDF if it contains multiple pages
                    if len(invoice["images"]) > 0:
                        temp_dir = os.path.join(os.getcwd(), "temp_invoices")
                        os.makedirs(temp_dir, exist_ok=True)
                        temp_pdf_path = os.path.join(temp_dir, f"temp_invoice_{i+1}.pdf")
                        
                        try:
                            invoice["images"][0].save(
                                temp_pdf_path, 
                                save_all=True, 
                                append_images=invoice["images"][1:] if len(invoice["images"]) > 1 else []
                            )
                            result["temp_file"] = temp_pdf_path
                        except Exception as e:
                            logger.error(f"Error saving temporary invoice PDF: {e}")
                    
                    results.append(result)
            else:
                # Single-page or single-invoice PDF
                extracted_text = extract_text_from_pdf(file_path)
                if not extracted_text.strip():
                    logger.warning(f"No text extracted from {file_path}")
                    results.append({"error": "No text extracted from file", "original_file": file_path})
                else:
                    # Query OpenAI to extract invoice information
                    logger.info(f"Extracted {len(extracted_text)} characters of text. Sending to OpenAI...")
                    result = query_llm(extracted_text)
                    
                    # Add detailed logging of the result
                    logger.info(f"OpenAI result for {os.path.basename(file_path)}: {result}")
                    
                    # Ensure result is a dictionary
                    if not isinstance(result, dict):
                        logger.warning(f"OpenAI returned unexpected result type: {type(result)}")
                        result = {"error": "Invalid response format from OpenAI", "original_file": file_path}
                    else:
                        result["original_file"] = file_path
                    results.append(result)  # Actually add the result to the list
        elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']:
            # Process a single image file
            extracted_text = extract_text_from_image(file_path)
            if not extracted_text.strip():
                logger.warning(f"No text extracted from {file_path}")
                results.append({"error": "No text extracted from file", "original_file": file_path})
            else:
                # Query OpenAI to extract invoice information
                logger.info(f"Extracted {len(extracted_text)} characters of text. Sending to OpenAI...")
                result = query_llm(extracted_text)
                
                # Ensure result is a dictionary
                if not isinstance(result, dict):
                    logger.warning(f"OpenAI returned unexpected result type: {type(result)}")
                    result = {"error": "Invalid response format from OpenAI", "original_file": file_path}
                else:
                    result["original_file"] = file_path
        else:
            # Unsupported file type
            logger.warning(f"Unsupported file type: {file_ext}")
            results.append({"error": f"Unsupported file type: {file_ext}", "original_file": file_path})
    
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        results.append({"error": f"Error processing file: {str(e)}", "original_file": file_path})
    
    return results

def save_and_move_invoice(result: Dict[str, Any]) -> None:
    """
    Rename and move a processed invoice based on extracted data.
    """
    original_file = result.get("original_file")
    
    if not original_file:
        logger.error("Missing original file information")
        return
    
    # Determine if we're working with a split invoice or the original file
    source_file = result.get("temp_file", original_file)
    
    # Generate the new filename based on extracted data
    new_filename, is_complete = generate_invoice_filename(result, os.path.basename(original_file))
    
    # Create the destination directories if they don't exist
    complete_dir = os.path.join(os.getcwd(), get_directory_path("complete"))
    incomplete_dir = os.path.join(os.getcwd(), get_directory_path("incomplete"))
    
    os.makedirs(complete_dir, exist_ok=True)
    os.makedirs(incomplete_dir, exist_ok=True)
    
    # Determine the destination path
    dest_dir = complete_dir if is_complete else incomplete_dir
    dest_path = os.path.join(dest_dir, new_filename)
    
    try:
        # Copy/move the file to destination
        if os.path.exists(source_file):
            # Copy the file to its new location
            shutil.copy2(source_file, dest_path)
            
            # If this is a split invoice (we made a temporary file), we don't delete the original
            if result.get("is_split", False) and "temp_file" in result:
                logger.info(f"Moved split invoice to: {dest_path}")
                
                # Clean up temporary file if it's not the original
                try:
                    if source_file != original_file:
                        os.remove(source_file)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file {source_file}: {e}")
            else:
                # If it's an original file, we've copied it, so now we can remove the original
                logger.info(f"Moved file to: {dest_path}")
                # Move original file to Backup folder instead of deleting it
                if os.path.abspath(original_file) != os.path.abspath(dest_path):
                    try:
                        backup_dir = os.path.join(os.getcwd(), get_directory_path("backup"))
                        backup_path = os.path.join(backup_dir, os.path.basename(original_file))
                        
                        # Create a unique filename if a file with the same name already exists in Backup
                        if os.path.exists(backup_path):
                            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                            backup_filename = f"{os.path.splitext(os.path.basename(original_file))[0]}_{timestamp}{os.path.splitext(original_file)[1]}"
                            backup_path = os.path.join(backup_dir, backup_filename)
                        
                        shutil.move(original_file, backup_path)
                        logger.info(f"Moved original file to backup: {backup_path}")
                    except Exception as e:
                        logger.warning(f"Failed to move original file to backup: {e}")
            
            return True
        else:
            logger.error(f"Source file not found: {source_file}")
            return False
    except Exception as e:
        logger.error(f"Error moving invoice file: {e}")
        return False

def main():
    """
    Main function to process all invoice files in the directory.
    
    Command-line options:
        --aggressive-split    : Treat each page as a separate invoice (default: off)
        --conservative-split  : Treat consecutive pages as parts of the same invoice if they appear related (default: on)
        --threshold=VALUE     : Set the similarity threshold for determining if pages belong to the same invoice (0.0-1.0)
                                Lower values (e.g., 0.3-0.5) will group more pages together
                                Higher values (e.g., 0.7-0.9) will require stronger similarity to group pages
    """
    # Declare global variables at the beginning of the function
    global PROCESSING_MODE
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process invoice documents and extract information.')
    parser.add_argument('--aggressive-split', action='store_true', help='Treat each page as a separate invoice')
    parser.add_argument('--conservative-split', action='store_true', help='Keep pages together when they appear related')
    parser.add_argument('--threshold', type=float, help='Set similarity threshold (0.0-1.0)')
    parser.add_argument('--pdf-as-invoice', action='store_true', help='Treat each PDF file as a single invoice')
    parser.add_argument('--bundle-mode', action='store_true', help='Allow multiple invoices within a single PDF')
    
    # Support both the legacy way (checking sys.argv) and the new argparse way
    args, unknown = parser.parse_known_args()
    
    # Configure invoice splitting behavior
    if args.aggressive_split or '--aggressive-split' in sys.argv:
        os.environ['AGGRESSIVE_INVOICE_SPLIT'] = 'true'
        logger.info("Using aggressive splitting mode: Each page will be treated as a separate invoice")
    elif args.conservative_split or '--conservative-split' in sys.argv:
        os.environ['AGGRESSIVE_INVOICE_SPLIT'] = 'false'
        logger.info("Using conservative splitting mode: Pages will be kept together when they appear to be part of the same invoice")
    else:
        logger.info("Using default conservative splitting mode: Pages will be kept together when they appear to be part of the same invoice")
    
    # Set custom threshold if provided
    if args.threshold is not None and 0.0 <= args.threshold <= 1.0:
        os.environ['INVOICE_SIMILARITY_THRESHOLD'] = str(args.threshold)
        logger.info(f"Using custom similarity threshold: {args.threshold}")
        logger.info(f"  (Lower values group more pages together, higher values require stronger similarity)")
    else:
        # Legacy way of setting threshold
        for arg in sys.argv:
            if arg.startswith('--threshold='):
                try:
                    threshold = float(arg.split('=')[1])
                    if 0.0 <= threshold <= 1.0:
                        os.environ['INVOICE_SIMILARITY_THRESHOLD'] = str(threshold)
                        logger.info(f"Using custom similarity threshold: {threshold}")
                        logger.info(f"  (Lower values group more pages together, higher values require stronger similarity)")
                except:
                    logger.warning(f"Invalid threshold value in {arg}. Must be a number between 0.0 and 1.0.")
    
    # Get processing mode and similarity threshold from config first
    global PROCESSING_MODE
    PROCESSING_MODE = get_processing_mode()  # From config.yaml
    similarity_threshold = get_similarity_threshold()  # From config.yaml
    
    # Set processing mode based on command-line arguments (overrides config)
    if args.pdf_as_invoice or '--pdf-as-invoice' in sys.argv:
        PROCESSING_MODE = 'PDF_AS_INVOICE'
        logger.info("Using PDF_AS_INVOICE mode: Each PDF file will be treated as a single invoice")
    elif args.bundle_mode or '--bundle-mode' in sys.argv:
        PROCESSING_MODE = 'BUNDLE'
        logger.info("Using BUNDLE mode: One PDF file can contain multiple invoices")
    else:
        logger.info(f"Using default processing mode: {PROCESSING_MODE}")

    # Ensure default similarity threshold is used if not specified
    if 'INVOICE_SIMILARITY_THRESHOLD' not in os.environ:
        os.environ['INVOICE_SIMILARITY_THRESHOLD'] = str(similarity_threshold)
        logger.info(f"Using default similarity threshold: {similarity_threshold}")
    api_key = os.environ.get('OPENAI_API_KEY', CONFIG.get("global", {}).get("openai_key", ""))
    if not api_key:
        logger.warning("No OpenAI API key found!")
        logger.warning("Please either:")
        logger.warning("1. Set the OPENAI_API_KEY environment variable, or")
        logger.warning("2. Update the openai_key in your config.yaml file")
        logger.warning("Exiting...")
        return
    else:
        # If using the hardcoded key, set it as an environment variable for the openai library
        if not os.environ.get('OPENAI_API_KEY'):
            os.environ['OPENAI_API_KEY'] = api_key
            logger.info("Using hardcoded OpenAI API key for invoice processing")
    
    logger.info("Using OpenAI API (GPT-4.1-mini) for invoice text processing")
    
    # Get all PDF and image files in the current directory
    # Get all PDF and image files in the current directory
    files = []
    
    # Define file patterns to look for
    file_patterns = ['*.pdf', '*.PDF', '*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG', '*.tif', '*.tiff', '*.TIF', '*.TIFF']
    
    # Use platform-specific path handling
    current_dir = os.getcwd()
    logger.info(f"Searching for files in: {current_dir}")
    
    for pattern in file_patterns:
        pattern_files = glob.glob(os.path.join(current_dir, pattern))
        files.extend(pattern_files)
    if not files:
        logger.info("No PDF or image files found in the current directory.")
        return
    
    logger.info(f"Found {len(files)} files to process.")
    
    # Create output directories if they don't exist using absolute paths for Windows compatibility
    os.makedirs(os.path.join(current_dir, "Rechnungen"), exist_ok=True)
    os.makedirs(os.path.join(current_dir, "Rechnungen-Unsicher"), exist_ok=True)
    os.makedirs(os.path.join(current_dir, "Backup"), exist_ok=True)  # Create Backup directory
    
    # On Windows, print some useful information
    if platform.system() == "Windows":
        logger.info("Running on Windows system")
        logger.info(f"Python version: {platform.python_version()}")
        logger.info(f"Tesseract command: {pytesseract.pytesseract.tesseract_cmd}")
        logger.info(f"Output directories set up in: {current_dir}")
    
    # Process each file
    successful_files = 0
    
    for file_path in files:
        try:
            # Process the file - may return multiple results for multi-invoice PDFs
            results = process_file(file_path)
            
            # Ensure results is always a list
            if results is None:
                logger.warning(f"No results were returned for {file_path}")
                results = []
            elif not isinstance(results, list):
                logger.warning(f"Unexpected result type for {file_path}, converting to list")
                results = [results]
                
            if not results:
                logger.warning(f"No results to process for {file_path}")
                # Add more detailed debugging
                logger.error(f"No valid invoice data was extracted from {file_path}. Check the OpenAI response for errors.")
                continue
            # Print and save each result
            for i, result in enumerate(results):
                invoice_label = f"Invoice {i+1} in " if len(results) > 1 else ""
                
                logger.info("\n" + "="*50)
                logger.info(f"Results for: {invoice_label}{file_path}")
                logger.info("="*50)
                
                error_msg = result.get("error", "")
                
                if error_msg and not any(key in result for key in ["invoice_date", "invoice_number", "company_name", "invoice_amount"]):
                    logger.warning(f"Error: {error_msg}")
                else:
                    logger.info(f"Invoice Date:   {result.get('invoice_date', 'Not found')}")
                    logger.info(f"Invoice Number: {result.get('invoice_number', 'Not found')}")
                    logger.info(f"Company Name:   {result.get('company_name', 'Not found')}")
                    logger.info(f"Invoice Amount: {result.get('invoice_amount', 'Not found')}")
                    
                    if error_msg:
                        logger.warning(f"\nWarning: {error_msg}")
                    
                    # Save and move the invoice
                    save_and_move_invoice(result)
                    successful_files += 1
            
            # Check if any results contain errors
            has_errors = False
            for result in results:
                if "error" in result:
                    has_errors = True
                    logger.warning(f"Found error in result: {result.get('error')}")
                    break
            
            # Get the keep_files_on_error setting from config
            keep_files_on_error = CONFIG.get("global", {}).get("error_handling", {}).get("keep_files_on_error", True)
            
            # Move the original file to Backup folder only if:
            # 1. No errors occurred, or
            # 2. keep_files_on_error is set to False (user wants to move files even if errors occurred)
            if os.path.exists(file_path) and (not has_errors or not keep_files_on_error):
                try:
                    backup_dir = os.path.join(os.getcwd(), get_directory_path("backup"))
                    backup_path = os.path.join(backup_dir, os.path.basename(file_path))
                    
                    # Create a unique filename if a file with the same name already exists in Backup
                    if os.path.exists(backup_path):
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        backup_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_{timestamp}{os.path.splitext(file_path)[1]}"
                        backup_path = os.path.join(backup_dir, backup_filename)
                    
                    shutil.move(file_path, backup_path)
                    logger.info(f"Moved original file to backup: {backup_path}")
                except Exception as e:
                    logger.error(f"Error moving original file to backup: {e}")
            elif has_errors and keep_files_on_error:
                # Log that we're keeping the file in place due to errors
                logger.info(f"Keeping file {file_path} in original location due to processing errors.")
        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    logger.info("\nProcessing complete.")
    logger.info(f"Processed {successful_files} invoices successfully.")
    
    # Clean up any temporary directories
    temp_dir = os.path.join(os.getcwd(), "temp_invoices")
    if os.path.exists(temp_dir):
        try:
            # Remove only if empty
            if not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            logger.warning(f"Could not remove temporary directory: {e}")

if __name__ == "__main__":
    main()

