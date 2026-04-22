import gradio as gr
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

LABELS = [
    'Baked Potato', 'Burger', 'Crispy Chicken', 'Donut', 'Fries', 
    'Hot Dog', 'Pizza', 'Sandwich', 'Taco', 'Taquito'
]

def render_tab_mosaic(model):
    def predict_mosaic(input_img):
        if input_img is None or model is None:
            return {"Lỗi hệ thống": 0.0}
        
        # Tiền xử lý ảnh giống như lúc huấn luyện
        raw_img = Image.fromarray(input_img).convert('RGB').resize((150, 150))
        
        # PyTorch transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            # Thông thường nếu dùng ResNet pretrained ImageNet thì có:
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(raw_img).unsqueeze(0) # Thêm batch dimension
        
        # Chuyển model tới đúng thiết bị
        device = next(model.parameters()).device
        img_tensor = img_tensor.to(device)
        
        # Suy luận
        model.eval()
        with torch.no_grad():
            outputs = model(img_tensor)
            # Dùng softmax để lấy xác suất
            probabilities = F.softmax(outputs, dim=1)[0]
            
        # Chuyển về numpy array
        probabilities = probabilities.cpu().numpy()
        
        return {LABELS[i]: float(probabilities[i]) for i in range(len(LABELS))}

    with gr.Row():
        with gr.Column(scale=1):
            src_image = gr.Image(label="Ảnh kiểm thử", type="numpy")
            submit_btn = gr.Button("Phân loại (Mosaic)", variant="primary")
            
        with gr.Column(scale=1):
            result_display = gr.Label(label="Kết quả mô hình Mosaic", num_top_classes=3)

    submit_btn.click(
        fn=predict_mosaic, 
        inputs=src_image, 
        outputs=result_display
    )
