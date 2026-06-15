import gradio as gr
import os
import zipfile
import tempfile
import random
import numpy as np
from PIL import Image, ImageFilter
import torchvision.transforms as T
from aug_cutmix import get_cutmix_bbox
from aug_mosaic import pil_mosaic

def apply_single_augmentation(img, option, bg_images):
    """Áp dụng 1 hiệu ứng duy nhất (Dành cho Demo)"""
    if option == "Lật ngang (Horizontal Flip)":
        return img.transpose(Image.FLIP_LEFT_RIGHT)
        
    elif option == "Xoay ảnh ngẫu nhiên (Rotation)":
        angle = random.choice([random.uniform(-45, -20), random.uniform(20, 45)])
        return img.rotate(angle, resample=Image.BICUBIC)
        
    elif option == "Phóng to/Thu nhỏ (Zoom/Crop)":
        zoom_factor = random.choice([random.uniform(0.7, 0.8), random.uniform(1.2, 1.3)])
        w, h = img.size
        if zoom_factor > 1.0:
            nw, nh = int(w / zoom_factor), int(h / zoom_factor)
            left = (w - nw) // 2
            top = (h - nh) // 2
            return img.crop((left, top, left + nw, top + nh)).resize((w, h), Image.Resampling.LANCZOS)
        else:
            nw, nh = int(w * zoom_factor), int(h * zoom_factor)
            resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
            new_img = Image.new("RGB", (w, h), (0, 0, 0))
            left = (w - nw) // 2
            top = (h - nh) // 2
            new_img.paste(resized, (left, top))
            return new_img

    elif option == "Chỉnh màu sắc, độ sáng (Color Jitter)":
        jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        return jitter(img)
        
    elif option == "Làm mờ (Gaussian Blur)":
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(2.0, 3.0)))
        
    elif option == "Trộn ảnh CutMix (CutMix)":
        if bg_images:
            bg_img = random.choice(bg_images).resize(img.size, Image.Resampling.LANCZOS)
            lam = np.random.beta(1.0, 1.0)
            bbx1, bby1, bbx2, bby2 = get_cutmix_bbox(img.size, lam)
            if bbx2 > bbx1 and bby2 > bby1:
                res_img = img.copy()
                region = bg_img.crop((bbx1, bby1, bbx2, bby2))
                res_img.paste(region, (bbx1, bby1))
                return res_img
        return img
            
    elif option == "Ghép ảnh khảm (Mosaic)":
        if bg_images:
            return pil_mosaic(img, bg_images, size=img.size)
        return img
        
    return img


def apply_pipeline_augmentation(img, options, bg_images):
    """Trộn ngẫu nhiên các hiệu ứng (Dành cho Training)"""
    if "Lật ngang (Horizontal Flip)" in options and random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
    if "Xoay ảnh ngẫu nhiên (Rotation)" in options:
        angle = random.uniform(-30, 30)
        img = img.rotate(angle, resample=Image.BICUBIC)
        
    if "Phóng to/Thu nhỏ (Zoom/Crop)" in options:
        zoom_factor = random.uniform(0.8, 1.2)
        w, h = img.size
        if zoom_factor > 1.0:
            nw, nh = int(w / zoom_factor), int(h / zoom_factor)
            left = (w - nw) // 2
            top = (h - nh) // 2
            img = img.crop((left, top, left + nw, top + nh)).resize((w, h), Image.Resampling.LANCZOS)
        else:
            nw, nh = int(w * zoom_factor), int(h * zoom_factor)
            resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
            new_img = Image.new("RGB", (w, h), (0, 0, 0))
            left = (w - nw) // 2
            top = (h - nh) // 2
            new_img.paste(resized, (left, top))
            img = new_img

    if "Chỉnh màu sắc, độ sáng (Color Jitter)" in options:
        jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        img = jitter(img)
        
    if "Làm mờ (Gaussian Blur)" in options and random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 3.0)))
        
    if "Trộn ảnh CutMix (CutMix)" in options and bg_images and random.random() < 0.5:
        bg_img = random.choice(bg_images).resize(img.size, Image.Resampling.LANCZOS)
        lam = np.random.beta(1.0, 1.0)
        bbx1, bby1, bbx2, bby2 = get_cutmix_bbox(img.size, lam)
        if bbx2 > bbx1 and bby2 > bby1:
            res_img = img.copy()
            region = bg_img.crop((bbx1, bby1, bbx2, bby2))
            res_img.paste(region, (bbx1, bby1))
            img = res_img
            
    if "Ghép ảnh khảm (Mosaic)" in options and bg_images and random.random() < 0.5:
        img = pil_mosaic(img, bg_images, size=img.size)
        
    return img


def handle_batch_augmentation(files, aug_options, mode, multiplier):
    if not files or not aug_options:
        return None
        
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "Augmented_Dataset.zip")
    
    bg_images = []
    if "Trộn ảnh CutMix (CutMix)" in aug_options or "Ghép ảnh khảm (Mosaic)" in aug_options:
        for file_obj in files:
            p = file_obj if isinstance(file_obj, str) else file_obj.name
            try:
                bg_images.append(Image.open(p).convert("RGB"))
            except:
                pass
                
    suffix_map = {
        "Lật ngang (Horizontal Flip)": "flip",
        "Xoay ảnh ngẫu nhiên (Rotation)": "rotate",
        "Phóng to/Thu nhỏ (Zoom/Crop)": "zoom",
        "Chỉnh màu sắc, độ sáng (Color Jitter)": "color",
        "Làm mờ (Gaussian Blur)": "blur",
        "Trộn ảnh CutMix (CutMix)": "cutmix",
        "Ghép ảnh khảm (Mosaic)": "mosaic"
    }

    is_training_mode = "Trộn" in mode

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file_obj in files:
            path = file_obj if isinstance(file_obj, str) else file_obj.name
            filename = os.path.basename(path)
            name, ext = os.path.splitext(filename)
            if not ext:
                ext = ".jpg"
                
            try:
                img = Image.open(path).convert("RGB")
                
                orig_name = f"{name}_orig{ext}"
                orig_path = os.path.join(temp_dir, orig_name)
                img.save(orig_path)
                zipf.write(orig_path, orig_name)
                
                if is_training_mode:
                    for i in range(int(multiplier)):
                        aug_img = apply_pipeline_augmentation(img.copy(), aug_options, bg_images)
                        aug_name = f"{name}_aug_mix_{i+1}{ext}"
                        aug_path = os.path.join(temp_dir, aug_name)
                        aug_img.save(aug_path)
                        zipf.write(aug_path, aug_name)
                else:
                    for option in aug_options:
                        aug_img = apply_single_augmentation(img.copy(), option, bg_images)
                        suffix = suffix_map.get(option, 'aug')
                        aug_name = f"{name}_aug_{suffix}{ext}"
                        aug_path = os.path.join(temp_dir, aug_name)
                        aug_img.save(aug_path)
                        zipf.write(aug_path, aug_name)
                    
            except Exception as e:
                print(f"Lỗi xử lý ảnh {filename}: {e}")
                
    return zip_path

def render_tab_batch_augmentation():
    gr.Markdown("""
    ## Trạm Tăng Cường Dữ Liệu (Data Augmentation Station)
    Tính năng này giúp bạn nhân bản và làm phong phú dữ liệu (Data Augmentation).
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            file_uploader = gr.File(
                label="1. Tải lên danh sách ảnh gốc",
                file_count="multiple",
                type="filepath"
            )
            
            aug_options = gr.CheckboxGroup(
                choices=[
                    "Lật ngang (Horizontal Flip)",
                    "Xoay ảnh ngẫu nhiên (Rotation)",
                    "Phóng to/Thu nhỏ (Zoom/Crop)",
                    "Chỉnh màu sắc, độ sáng (Color Jitter)",
                    "Làm mờ (Gaussian Blur)",
                    "Trộn ảnh CutMix (CutMix)",
                    "Ghép ảnh khảm (Mosaic)"
                ],
                value=[
                    "Lật ngang (Horizontal Flip)",
                    "Xoay ảnh ngẫu nhiên (Rotation)",
                    "Phóng to/Thu nhỏ (Zoom/Crop)",
                    "Ghép ảnh khảm (Mosaic)"
                ],
                label="2. Chọn các hiệu ứng (Pool of augmentations)"
            )
            
            aug_mode = gr.Radio(
                choices=["Trộn ngẫu nhiên (Pipeline Stacking - Tốt nhất để Train AI)", "Tách lẻ từng hiệu ứng (Dễ nhìn - Dành cho Demo)"],
                value="Trộn ngẫu nhiên (Pipeline Stacking - Tốt nhất để Train AI)",
                label="3. Chế độ sinh dữ liệu"
            )
            
            multiplier_slider = gr.Slider(
                minimum=1, maximum=20, value=5, step=1,
                label="4. Số bản sao sinh ra TỪ MỖI ảnh gốc (Chỉ dùng cho chế độ Trộn)"
            )
            
            run_btn = gr.Button("Chạy Tăng Cường (Augment Data)", variant="primary")
            
        with gr.Column(scale=1):
            download_output = gr.File(label="Kết quả (Tải file Zip tại đây)")
            
    run_btn.click(
        fn=handle_batch_augmentation,
        inputs=[file_uploader, aug_options, aug_mode, multiplier_slider],
        outputs=download_output
    )
