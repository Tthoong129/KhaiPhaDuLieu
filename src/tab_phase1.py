import gradio as gr
import numpy as np
from PIL import Image

LABELS = [
    'Baked Potato', 'Burger', 'Crispy Chicken', 'Donut', 'Fries', 
    'Hot Dog', 'Pizza', 'Sandwich', 'Taco', 'Taquito'
]

def render_tab_phase1(model):
    def handle_prediction(input_img):
        if model is None:
            return {"Lỗi: Mô hình Baseline chưa được nạp": 0.0}
        if input_img is None:
            return {"Lỗi: Vui lòng tải ảnh lên trước khi phân loại": 0.0}
        try:
            processed_img = Image.fromarray(input_img).resize((150, 150))
            img_array = np.expand_dims(np.array(processed_img), axis=0)
            predictions = model.predict(img_array, verbose=0)[0]
            return {LABELS[i]: float(predictions[i]) for i in range(len(LABELS))}
        except Exception as e:
            return {f"Lỗi suy luận: {str(e)}": 0.0}

    with gr.Row():
        with gr.Column(scale=1):
            src_image = gr.Image(label="Ảnh đầu vào (Gốc)", type="numpy")
            predict_btn = gr.Button("Phân loại (Baseline)", variant="primary")
            
        with gr.Column(scale=1):
            result_label = gr.Label(label="Top 3 dự đoán", num_top_classes=3)

    predict_btn.click(
        fn=handle_prediction, 
        inputs=src_image, 
        outputs=result_label
    )