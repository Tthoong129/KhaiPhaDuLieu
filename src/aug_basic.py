import random
from PIL import Image

def pil_basic_augment(img, rotation_range=0.15, zoom_range=0.1, flip_h=True, flip_v=False):
    if flip_h and random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if flip_v and random.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        
    if rotation_range > 0:
        angle = random.uniform(-rotation_range * 180, rotation_range * 180)
        img = img.rotate(angle, resample=Image.BICUBIC)
        
    if zoom_range > 0:
        zoom_factor = random.uniform(1.0 - zoom_range, 1.0 + zoom_range)
        w, h = img.size
        if zoom_factor > 1.0:
            nw, nh = int(w / zoom_factor), int(h / zoom_factor)
            left = (w - nw) // 2
            top = (h - nh) // 2
            img = img.crop((left, top, left + nw, top + nh)).resize((w, h), Image.Resampling.LANCZOS)
        elif zoom_factor < 1.0:
            nw, nh = int(w * zoom_factor), int(h * zoom_factor)
            resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
            new_img = Image.new("RGB", (w, h), (0, 0, 0))
            left = (w - nw) // 2
            top = (h - nh) // 2
            new_img.paste(resized, (left, top))
            img = new_img
            
    return img
