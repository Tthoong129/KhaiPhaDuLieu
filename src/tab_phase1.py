import gradio as gr
import numpy as np
from PIL import Image

LABELS = ['Baked Potato', 'Burger', 'Crispy Chicken', 'Donut', 'Fries', 
          'Hot Dog', 'Pizza', 'Sandwich', 'Taco', 'Taquito']

def render_tab_phase1(model):
    def predict(img):
        if img is None or model is None: return {"Lỗi": 0.0}
        img_array = np.expand_dims(np.array(Image.fromarray(img).resize((150, 150))), axis=0)
        preds = model.predict(img_array, verbose=0)[0]
        return {LABELS[i]: float(preds[i]) for i in range(10)}

    gr.Markdown("Mô hình cơ bản (Baseline CNN) sử dụng dữ liệu gốc không qua tăng cường.")
    with gr.Row():
        img_input = gr.Image(label="Ảnh đầu vào")
        res_output = gr.Label(label="Kết quả dự đoán", num_top_classes=3)
    btn = gr.Button("Phân loại (Phase 1)")
    
    btn.click(fn=predict, inputs=img_input, outputs=res_output)