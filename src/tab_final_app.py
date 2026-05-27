import gradio as gr
import numpy as np
from PIL import Image
from sklearn.preprocessing import normalize

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F


LABELS = [
    "Baked Potato", "Burger", "Crispy Chicken", "Donut", "Fries",
    "Hot Dog", "Pizza", "Sandwich", "Taco", "Taquito"
]
CLUSTER_INFO = {
    0: {
        "main_label": "Burger",
        "main_count": 211,
        "total_images": 658,
        "percent": 32.07
    },
    1: {
        "main_label": "Hot Dog",
        "main_count": 368,
        "total_images": 1873,
        "percent": 19.65
    },
    2: {
        "main_label": "Baked Potato",
        "main_count": 286,
        "total_images": 1273,
        "percent": 22.47
    },
    3: {
        "main_label": "Pizza",
        "main_count": 609,
        "total_images": 2394,
        "percent": 25.44
    },
    4: {
        "main_label": "Taquito",
        "main_count": 481,
        "total_images": 3008,
        "percent": 15.99
    },
    5: {
        "main_label": "Donut",
        "main_count": 400,
        "total_images": 1051,
        "percent": 38.06
    },
    6: {
        "main_label": "Donut",
        "main_count": 575,
        "total_images": 1438,
        "percent": 39.99
    },
    7: {
        "main_label": "Donut",
        "main_count": 188,
        "total_images": 912,
        "percent": 20.61
    },
    8: {
        "main_label": "Fries",
        "main_count": 347,
        "total_images": 1236,
        "percent": 28.07
    },
    9: {
        "main_label": "Taco",
        "main_count": 249,
        "total_images": 1157,
        "percent": 21.52
    }
}

def predict_cluster(input_img, kmeans_model, feature_extractor):
    if input_img is None:
        return "Chưa có ảnh đầu vào"

    if kmeans_model is None:
        return "Chưa nạp K-Means model"

    if feature_extractor is None:
        return "Chưa có feature extractor từ model Phase 1"

    try:
        img = Image.fromarray(input_img).convert("RGB").resize((150, 150))
        img_array = np.expand_dims(np.array(img), axis=0)

        feature_vector = feature_extractor.predict(img_array, verbose=0)

        if hasattr(kmeans_model, "n_features_in_"):
            expected_dim = kmeans_model.n_features_in_
            actual_dim = feature_vector.shape[1]

            if actual_dim != expected_dim:
                return (
                    f"Lỗi: Feature vector có {actual_dim} chiều, "
                    f"nhưng K-Means cần {expected_dim} chiều"
                )
        feature_vector_norm = normalize(feature_vector)
        cluster_id = int(kmeans_model.predict(feature_vector_norm)[0])

        info = CLUSTER_INFO.get(cluster_id)

        if info is None:
            return f"Ảnh này thuộc Cluster {cluster_id}: chưa có thống kê mô tả cụm."

        main_label = info["main_label"]
        main_count = info["main_count"]
        total_images = info["total_images"]
        percent = info["percent"]

        return (
            f"Ảnh này thuộc Cluster {cluster_id}: "
            f"nhóm đặc trưng pha trộn, nhãn xuất hiện nhiều nhất là {main_label} "
            f"({main_count}/{total_images} ảnh - {percent}%)."
        )
    except Exception as e:
        return f"Lỗi dự đoán cluster: {e}"


def preprocess_for_pytorch(input_img):
    img = Image.fromarray(input_img).convert("RGB").resize((150, 150))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transform(img).unsqueeze(0)


def predict_pytorch_model(input_img, model, model_name):
    if model is None:
        return {f"Lỗi: Model {model_name} chưa được nạp": 0.0}, ""

    try:
        img_tensor = preprocess_for_pytorch(input_img)
        device = next(model.parameters()).device
        img_tensor = img_tensor.to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)[0].cpu().numpy()

        result = {
            LABELS[i]: float(probabilities[i])
            for i in range(len(LABELS))
        }

        top_idx = int(np.argmax(probabilities))
        summary = (
            f"Model {model_name} dự đoán: "
            f"{LABELS[top_idx]} ({probabilities[top_idx] * 100:.2f}%)"
        )

        return result, summary

    except Exception as e:
        return {f"Lỗi suy luận {model_name}: {e}": 0.0}, ""


def predict_keras_model(input_img, model, model_name):
    if model is None:
        return {f"Lỗi: Model {model_name} chưa được nạp": 0.0}, ""

    try:
        img = Image.fromarray(input_img).convert("RGB").resize((150, 150))
        img_array = np.expand_dims(np.array(img), axis=0)

        probabilities = model.predict(img_array, verbose=0)[0]

        result = {
            LABELS[i]: float(probabilities[i])
            for i in range(len(LABELS))
        }

        top_idx = int(np.argmax(probabilities))
        summary = (
            f"Model {model_name} dự đoán: "
            f"{LABELS[top_idx]} ({probabilities[top_idx] * 100:.2f}%)"
        )

        return result, summary

    except Exception as e:
        return {f"Lỗi suy luận {model_name}: {e}": 0.0}, ""


def render_tab_final_app(
    model_p1,
    model_p2,
    model_p3,
    model_mosaic,
    kmeans_model,
    feature_extractor
):
    gr.Markdown("""
    ## Final Application - Ứng dụng phân loại thực phẩm

    Tab này tích hợp model phân loại và K-Means Clustering.
    Người dùng upload ảnh, chọn mô hình dự đoán, sau đó hệ thống trả về Top 3 kết quả và ảnh thuộc cluster nào.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Ảnh đầu vào", type="numpy")

            model_choice = gr.Radio(
                choices=["Baseline", "Augmented", "CutMix", "Mosaic"],
                value="CutMix",
                label="Chọn mô hình phân loại"
            )

            predict_btn = gr.Button("Dự đoán", variant="primary")

        with gr.Column(scale=1):
            result_label = gr.Label(label="Top 3 dự đoán", num_top_classes=3)
            summary_box = gr.Textbox(label="Kết luận dự đoán")
            cluster_box = gr.Textbox(label="Thông tin Cluster")

    def handle_predict(input_img, model_choice):
        if input_img is None:
            return (
                {"Lỗi: Vui lòng tải ảnh lên trước": 0.0},
                "Chưa có ảnh đầu vào",
                "Chưa có ảnh đầu vào"
            )

        if model_choice == "Baseline":
            result, summary = predict_keras_model(input_img, model_p1, "Baseline")
        elif model_choice == "Augmented":
            result, summary = predict_keras_model(input_img, model_p2, "Augmented")
        elif model_choice == "CutMix":
            result, summary = predict_pytorch_model(input_img, model_p3, "CutMix")
        else:
            result, summary = predict_pytorch_model(input_img, model_mosaic, "Mosaic")

        cluster_info = predict_cluster(input_img, kmeans_model, feature_extractor)

        return result, summary, cluster_info

    predict_btn.click(
        fn=handle_predict,
        inputs=[input_img, model_choice],
        outputs=[result_label, summary_box, cluster_box]
    )