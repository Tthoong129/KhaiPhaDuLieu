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

def render_tab_phase3(model):
    def predict_cutmix(input_img):
        if input_img is None or model is None:
            return {"Lỗi hệ thống": 0.0}
        
        # Tiền xử lý ảnh giống như lúc huấn luyện
        raw_img = Image.fromarray(input_img).convert('RGB').resize((150, 150))
        
        # PyTorch transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            # Nếu model được train với normalize, thêm vào đây. Chúng ta thử không thêm vì keras cũng không thấy ghi.
            # Hoặc dùng normalize cơ bản nếu cần. Tạm thời dùng ToTensor() (scale [0, 1]).
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
            submit_btn = gr.Button("Phân loại (CutMix)", variant="primary")
            
        with gr.Column(scale=1):
            result_display = gr.Label(label="Kết quả mô hình CutMix", num_top_classes=3)

    submit_btn.click(
        fn=predict_cutmix, 
        inputs=src_image, 
        outputs=result_display
    )
