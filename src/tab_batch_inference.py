import gradio as gr
import numpy as np
from PIL import Image
import os
import pandas as pd
import zipfile
import tempfile
from sklearn.preprocessing import normalize
from tab_final_app import predict_keras_model, predict_pytorch_model, CLUSTER_INFO

def handle_batch_inference(files, model_choice, model_p1, model_p2, model_p3, model_mosaic, kmeans_model, feature_extractor):
    if not files:
        df = pd.DataFrame(columns=["Tên ảnh", "Nhãn dự đoán", "Độ tự tin (Confidence Score)", "Cụm (K-Means)"])
        return df, []
    
    results = []
    state_data = []
    
    for file_obj in files:
        path = file_obj if isinstance(file_obj, str) else file_obj.name
        filename = os.path.basename(path)
        
        try:
            img = Image.open(path).convert("RGB")
            input_img = np.array(img)
            
            # 1. Chạy Classification Model
            if model_choice == "Baseline":
                res, sum_str = predict_keras_model(input_img, model_p1, "Baseline")
            elif model_choice == "Augmented":
                res, sum_str = predict_keras_model(input_img, model_p2, "Augmented")
            elif model_choice == "CutMix":
                res, sum_str = predict_pytorch_model(input_img, model_p3, "CutMix")
            else:
                res, sum_str = predict_pytorch_model(input_img, model_mosaic, "Mosaic")
            
            best_label = max(res, key=res.get)
            best_score = res[best_label]
            score_str = f"{best_score * 100:.2f}%"
            
            # 2. Chạy K-Means Clustering
            cluster_str = "N/A"
            if kmeans_model is not None and feature_extractor is not None:
                try:
                    img_resized = img.resize((150, 150))
                    img_array = np.expand_dims(np.array(img_resized), axis=0)
                    feature_vector = feature_extractor.predict(img_array, verbose=0)
                    feature_vector_norm = normalize(feature_vector)
                    cluster_id = int(kmeans_model.predict(feature_vector_norm)[0])
                    
                    info = CLUSTER_INFO.get(cluster_id)
                    if info:
                        cluster_str = f"Cụm {cluster_id} ({info['main_label']})"
                    else:
                        cluster_str = f"Cụm {cluster_id}"
                except Exception as kmeans_err:
                    cluster_str = "Lỗi K-Means"

            results.append([filename, best_label, score_str, cluster_str])
            state_data.append({"path": path, "filename": filename, "label": best_label})
        except Exception as e:
            results.append([filename, "Lỗi", str(e), "Lỗi"])
            state_data.append({"path": path, "filename": filename, "label": "Lỗi"})
            
    df = pd.DataFrame(results, columns=["Tên ảnh", "Nhãn dự đoán", "Độ tự tin (Confidence Score)", "Cụm (K-Means)"])
    return df, state_data

def handle_export(state_data):
    if not state_data:
        return None
        
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "ket_qua_phan_loai.zip")
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for item in state_data:
            label = item["label"]
            if label == "Lỗi":
                continue
            path = item["path"]
            filename = item["filename"]
            arcname = os.path.join(label, filename)
            zipf.write(path, arcname)
            
    return zip_path

def handle_select(evt: gr.SelectData, state_data):
    row_idx = evt.index[0]
    if state_data and row_idx < len(state_data):
        return state_data[row_idx]["path"]
    return None

def render_tab_batch_inference(model_p1, model_p2, model_p3, model_mosaic, kmeans_model, feature_extractor):
    with gr.Row(equal_height=False):
        # Cột Trái: Upload & Chạy (scale=1)
        with gr.Column(scale=1):
            # Card 1: Chọn ảnh đầu vào
            with gr.Column(elem_classes="custom-card"):
                gr.HTML("<div class='card-title'>1. Tải lên tập ảnh gốc</div>")
                file_uploader = gr.File(
                    label="Tải lên thư mục / nhiều ảnh",
                    file_count="multiple",
                    type="filepath",
                    height=100,
                    show_label=False,
                    elem_classes=["compact-file-uploader"]
                )
                
            # Card 2: Chọn mô hình phân loại
            with gr.Column(elem_classes="custom-card"):
                gr.HTML("<div class='card-title'>2. Chọn mô hình</div>")
                model_choice = gr.Radio(
                    choices=["Baseline", "Augmented", "CutMix", "Mosaic"],
                    value="Mosaic",
                    show_label=False
                )
                
            # Card 3: Thao tác & Kết quả zip
            with gr.Column(elem_classes="custom-card"):
                gr.HTML("<div class='card-title'>3. Hành động & Xuất bản</div>")
                predict_btn = gr.Button("🚀 Xử lý hàng loạt (Batch Run)", variant="primary")
                export_btn = gr.Button("🗂️ Xuất & Tải về (Chia tệp)", variant="secondary")
                download_output = gr.File(label="Tải file Zip kết quả", height=70, elem_classes=["compact-file-downloader"])
            
        # Cột Phải: Xem bảng kết quả & Chi tiết ảnh
        with gr.Column(scale=2):
            # Card 4: Bảng kết quả phân loại
            with gr.Column(elem_classes="custom-card"):
                gr.HTML("<div class='card-title'>Bảng kết quả phân loại</div>")
                output_table = gr.Dataframe(
                    headers=["Tên ảnh", "Nhãn dự đoán", "Độ tự tin (Confidence Score)", "Cụm (K-Means)"],
                    show_label=False,
                    interactive=False
                )
                gr.HTML("<div style='font-size: 0.825rem; color: #64748b; text-align: center; margin-top: 4px;'>💡 Bấm vào một dòng bất kỳ trên bảng để xem chi tiết ảnh tương ứng ở dưới.</div>")
                
            # Card 5: Ảnh được chọn xem chi tiết
            with gr.Column(elem_classes="custom-card"):
                gr.HTML("<div class='card-title'>Chi tiết ảnh được chọn</div>")
                selected_image = gr.Image(label="Ảnh được chọn", type="filepath", height=150, show_label=False, elem_classes=["compact-image-preview"])
                state_results = gr.State([])
            
    def predict_wrapper(files, choice):
        return handle_batch_inference(files, choice, model_p1, model_p2, model_p3, model_mosaic, kmeans_model, feature_extractor)
        
    predict_btn.click(
        fn=predict_wrapper,
        inputs=[file_uploader, model_choice],
        outputs=[output_table, state_results]
    )
    
    export_btn.click(
        fn=handle_export,
        inputs=state_results,
        outputs=download_output
    )
    
    output_table.select(
        fn=handle_select,
        inputs=state_results,
        outputs=selected_image
    )
