import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

def render_tab_augmentation(aug_tool):
    def demo_aug(img):
        if img is None or aug_tool is None: return []
        img_array = np.expand_dims(np.array(Image.fromarray(img).resize((150, 150))), axis=0)
        
        results = []
        for _ in range(4):
            aug_img = aug_tool(img_array, training=True)[0]
            results.append(tf.cast(aug_img, tf.uint8).numpy())
        return results

    gr.Markdown("Minh họa thuật toán Tăng cường dữ liệu (Data Augmentation) bằng cách áp dụng lật ngang, xoay ngẫu nhiên và thu phóng.")
    with gr.Row():
        img_input = gr.Image(label="Dữ liệu gốc")
        aug_gallery = gr.Gallery(label="Dữ liệu sinh ra (Batch = 4)", columns=2)
    # Nút primary sẽ có màu xanh trời
    btn = gr.Button("Thực thi Module", variant="primary")
    
    btn.click(fn=demo_aug, inputs=img_input, outputs=aug_gallery)