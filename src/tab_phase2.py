import gradio as gr
import numpy as np
from PIL import Image

LABELS = [
    'Baked Potato', 'Burger', 'Crispy Chicken', 'Donut', 'Fries', 
    'Hot Dog', 'Pizza', 'Sandwich', 'Taco', 'Taquito'
]

def render_tab_phase2(model):
    def predict_advanced(input_img):
        if input_img is None or model is None:
            return {"Lỗi hệ thống": 0.0}
        raw_img = Image.fromarray(input_img).resize((150, 150))
        img_array = np.expand_dims(np.array(raw_img), axis=0)
        
        predictions = model.predict(img_array, verbose=0)[0]
        return {LABELS[i]: float(predictions[i]) for i in range(len(LABELS))}

    with gr.Row():
        with gr.Column(scale=1):
            src_image = gr.Image(label="Ảnh kiểm thử", type="numpy")
            submit_btn = gr.Button("Phân loại (Advanced)", variant="primary")
            
        with gr.Column(scale=1):
            result_display = gr.Label(label="Kết quả mô hình Nâng cao", num_top_classes=3)

    submit_btn.click(
        fn=predict_advanced, 
        inputs=src_image, 
        outputs=result_display
    )