import random
from PIL import Image

def pil_mosaic(img, bg_images, center_scale=0.5, size=(224, 224)):
    W, H = size
    cx = int(W * center_scale)
    cy = int(H * center_scale)
    
    imgs = [img]
    chosen_bgs = random.sample(bg_images, min(len(bg_images), 3))
    while len(chosen_bgs) < 3:
        chosen_bgs.append(img.copy())
    imgs.extend(chosen_bgs)
    random.shuffle(imgs)
    
    canvas = Image.new("RGB", (W, H))
    
    im0 = imgs[0].resize((cx, cy), Image.Resampling.LANCZOS)
    canvas.paste(im0, (0, 0))
    
    im1 = imgs[1].resize((W - cx, cy), Image.Resampling.LANCZOS)
    canvas.paste(im1, (cx, 0))
    
    im2 = imgs[2].resize((cx, H - cy), Image.Resampling.LANCZOS)
    canvas.paste(im2, (0, cy))
    
    im3 = imgs[3].resize((W - cx, H - cy), Image.Resampling.LANCZOS)
    canvas.paste(im3, (cx, cy))
    
    return canvas
