import gradio as gr
import numpy as np
from PIL import Image

from aug_basic import pil_basic_augment
from aug_cutmix import get_cutmix_bbox
from aug_mosaic import pil_mosaic
from aug_helpers import get_bg_images, get_selected_bg

def render_tab_augmentation(aug_tool=None):
    def process_images(input_img, method, rotation, zoom, flips, beta, bg_option, center_scale):
        if input_img is None:
            return []
        try:
            pil_img = Image.fromarray(input_img).convert("RGB")
            pil_img = pil_img.resize((224, 224), Image.Resampling.LANCZOS)
            
            generated_results = []
            
            if method == "Basic Augmentation":
                flip_h = "Horizontal" in flips
                flip_v = "Vertical" in flips
                for _ in range(4):
                    aug_img = pil_basic_augment(pil_img, rotation, zoom, flip_h, flip_v)
                    generated_results.append(np.array(aug_img))
            
            elif method == "CutMix":
                for _ in range(4):
                    bg_img = get_selected_bg(bg_option)
                    if bg_img is None:
                        bg_img = pil_img.copy()
                    bg_img = bg_img.resize((224, 224), Image.Resampling.LANCZOS)
                    
                    lam = np.random.beta(beta, beta)
                    bbx1, bby1, bbx2, bby2 = get_cutmix_bbox((224, 224), lam)
                    
                    patch = bg_img.crop((bbx1, bby1, bbx2, bby2))
                    img_copy = pil_img.copy()
                    img_copy.paste(patch, (bbx1, bby1))
                    
                    generated_results.append(np.array(img_copy))
                    
            elif method == "Mosaic":
                bg_images = get_bg_images()
                if not bg_images:
                    bg_images = [pil_img.copy()]
                
                for _ in range(4):
                    mos_img = pil_mosaic(pil_img, bg_images, center_scale, size=(224, 224))
                    generated_results.append(np.array(mos_img))
            
            return generated_results
        except Exception as e:
            gr.Error(f"Lỗi khi thực hiện augmentation: {e}")
            return []

    def update_settings(method):
        return (
            gr.update(visible=(method == "Basic Augmentation")),
            gr.update(visible=(method == "CutMix")),
            gr.update(visible=(method == "Mosaic"))
        )

    with gr.Row(equal_height=False):
        # Cột Trái: Cấu hình
        with gr.Column(scale=1):
            # Card 1: Upload ảnh gốc
            with gr.Column(elem_classes="custom-card"):
                gr.HTML("<div class='card-title'>1. Chọn ảnh gốc</div>")
                src_image = gr.Image(label="Ảnh gốc", type="numpy", show_label=False, height=180, elem_classes=["compact-image-preview"])
                
            # Card 2: Chọn phương pháp tăng cường
            with gr.Column(elem_classes="custom-card"):
                gr.HTML("<div class='card-title'>2. Chọn phương pháp</div>")
                aug_method = gr.Radio(
                    choices=["Basic Augmentation", "CutMix", "Mosaic"], 
                    value="Basic Augmentation", 
                    show_label=False
                )
                
            # Card 3: Cấu hình tham số và Nút thực thi
            with gr.Column(elem_classes="custom-card"):
                gr.HTML("<div class='card-title'>3. Tham số tăng cường</div>")
                
                with gr.Group(visible=True) as basic_box:
                    rotation_slider = gr.Slider(minimum=0.0, maximum=0.5, value=0.15, step=0.05, label="Rotation")
                    zoom_slider = gr.Slider(minimum=0.0, maximum=0.5, value=0.1, step=0.05, label="Zoom")
                    flip_checkbox = gr.CheckboxGroup(choices=["Horizontal", "Vertical"], value=["Horizontal"], label="Flips")
                    
                with gr.Group(visible=False) as cutmix_box:
                    cutmix_beta = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Beta Parameter")
                    cutmix_bg = gr.Dropdown(
                        choices=["Random", "Fries", "Donut", "Hamburger", "Hotdog", "Pizza"], 
                        value="Random", 
                        label="Ảnh nền"
                    )
                    
                with gr.Group(visible=False) as mosaic_box:
                    mosaic_center = gr.Slider(minimum=0.2, maximum=0.8, value=0.5, step=0.05, label="Center Scale")
                
                gr.HTML("<div style='margin-top: 15px;'></div>")
                execute_btn = gr.Button("🚀 Bắt đầu sinh dữ liệu", variant="primary")
        
        # Cột Phải: Hiển thị kết quả
        with gr.Column(scale=2):
            with gr.Column(elem_classes="custom-card"):
                gr.HTML("<div class='card-title'>Kết quả tăng cường</div>")
                res_gallery = gr.Gallery(
                    label="Kết quả sinh dữ liệu", 
                    columns=2, 
                    rows=2,
                    height=200,
                    preview=True,
                    show_label=False,
                    elem_classes=["compact-gallery"]
                )
                gr.HTML("""
                <div style="font-size: 0.9rem; color: #64748b; text-align: center; margin-top: 15px;">
                    ✨ Kết quả tăng cường ảnh (4 mẫu sinh ra) sẽ hiển thị ở trên. Bạn có thể bấm chọn ảnh để xem chi tiết phóng to.
                </div>
                """)

    aug_method.change(
        fn=update_settings,
        inputs=aug_method,
        outputs=[basic_box, cutmix_box, mosaic_box]
    )

    execute_btn.click(
        fn=process_images, 
        inputs=[
            src_image, 
            aug_method, 
            rotation_slider, 
            zoom_slider, 
            flip_checkbox, 
            cutmix_beta, 
            cutmix_bg, 
            mosaic_center
        ], 
        outputs=res_gallery
    )