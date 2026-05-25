import os
import random
from PIL import Image

def get_bg_images():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    anh_test_dir = os.path.join(current_dir, "../AnhTest")
    bg_files = [
        "Fries.jpg",
        "donut.jpg",
        "hambuger.jpg",
        "hambuger2.jpg",
        "hotdog.jpg",
        "pizza.png"
    ]
    images = []
    for f in bg_files:
        path = os.path.join(anh_test_dir, f)
        if os.path.exists(path):
            try:
                images.append(Image.open(path).convert("RGB"))
            except Exception:
                pass
    return images

def get_selected_bg(option):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    anh_test_dir = os.path.join(current_dir, "../AnhTest")
    
    mapping = {
        "Fries": ["Fries.jpg"],
        "Donut": ["donut.jpg"],
        "Hamburger": ["hambuger.jpg", "hambuger2.jpg"],
        "Hotdog": ["hotdog.jpg"],
        "Pizza": ["pizza.png"]
    }
    
    if option == "Random" or option not in mapping:
        all_options = ["Fries.jpg", "donut.jpg", "hambuger.jpg", "hambuger2.jpg", "hotdog.jpg", "pizza.png"]
        choice = random.choice(all_options)
    else:
        choice = random.choice(mapping[option])
        
    path = os.path.join(anh_test_dir, choice)
    if os.path.exists(path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            pass
    return None
