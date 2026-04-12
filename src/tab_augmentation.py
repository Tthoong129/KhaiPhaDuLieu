import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

def render_tab_augmentation(aug_tool):
    def process_images(input_img):
        if input_img is None or aug_tool is None:
            return []
        raw_img = Image.fromarray(input_img).resize((150, 150))
        img_batch = np.expand_dims(np.array(raw_img), axis=0)
        
        generated_results = []
        for _ in range(4):
            aug_tensor = aug_tool(img_batch, training=True)[0]
            generated_results.append(tf.cast(aug_tensor, tf.uint8).numpy())
            
        return generated_results

    with gr.Row():
        with gr.Column(scale=1):
            src_image = gr.Image(label="Ảnh gốc", type="numpy")
            execute_btn = gr.Button("Gen Data", variant="primary")
        
        with gr.Column(scale=2):
            res_gallery = gr.Gallery(
                label="Kết quả Augmentation", 
                columns=2, 
                rows=2,
                height=400,
                preview=True
            )

    execute_btn.click(
        fn=process_images, 
        inputs=src_image, 
        outputs=res_gallery
    )