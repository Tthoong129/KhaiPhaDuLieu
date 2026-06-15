import os
import gradio as gr
import joblib
import numpy as np

try:
    import tensorflow as tf
    # Cấu hình để TensorFlow không chiếm hết VRAM GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as gpu_err:
            print(f"Canh bao: Khong the cau hinh GPU memory growth: {gpu_err}")
except Exception as e:
    tf = None
    print(f"Canh bao: Khong the nap thu vien TensorFlow/Keras. Cac tab Baseline, Augmented va Augmentation Tool se khong hoat dong. Chi tiet: {e}")

try:
    from augmentor import get_augmentation_model
except Exception as e:
    get_augmentation_model = None
    print(f"Canh bao: Khong the nap module augmentor. Chi tiet: {e}")

from tab_phase1 import render_tab_phase1
from tab_phase2 import render_tab_phase2
from tab_phase3 import render_tab_phase3
from tab_mosaic import render_tab_mosaic
from tab_augmentation import render_tab_augmentation
from tab_final_app import render_tab_final_app
from tab_batch_inference import render_tab_batch_inference
from tab_batch_augmentation import render_tab_batch_augmentation
import torch
import torchvision.models as models


current_dir = os.path.dirname(os.path.abspath(__file__))
path_p1 = os.path.join(current_dir, '../models/baseline_model_p1.keras')
path_p2 = os.path.join(current_dir, '../models/augmented_model_p2.keras')
path_p3 = os.path.join(current_dir, '../models/AI_FastFood_CutMix1.pth')
path_mosaic = os.path.join(current_dir, '../models/fastfood_resnet18_mosaic.pth')
path_kmeans = os.path.join(current_dir, '../models/kmeans_model.pkl')

model_p1 = None
model_p2 = None
aug_tool = None
model_p3 = None
model_mosaic = None
kmeans_model = None
feature_extractor = None

# Nạp model Phase 1
if tf is not None and os.path.exists(path_p1):
    try:
        model_p1 = tf.keras.models.load_model(path_p1)
        print("Da nap thanh cong mo hinh Phase 1: Baseline")
    except Exception as e:
        print(f"Loi nap mo hinh Phase 1: {e}")
else:
    print("Bo qua nap mo hinh Phase 1 (thieu file model hoac tensorflow)")

if model_p1 is not None and tf is not None:
    try:
        
        # Tạo feature extractor bằng cách bỏ lớp cuối softmax
        # Model gốc: ... -> Dense(128) -> Dense(10, softmax)
        # K-Means cần vector 128 chiều nên lấy output trước lớp cuối
        feature_extractor = tf.keras.Sequential(
            model_p1.layers[:-1],
            name="feature_extractor_p1"
        )

        # Gọi thử bằng ảnh giả để build model
        dummy_input = np.zeros((1, 150, 150, 3), dtype=np.float32)
        dummy_feature = feature_extractor.predict(dummy_input, verbose=0)

        print(f"Da tao feature extractor tu Phase 1, output shape: {dummy_feature.shape}")

    except Exception as e:
        print(f"Loi tao feature extractor tu Phase 1: {e}")


if os.path.exists(path_kmeans):
    try:
        kmeans_model = joblib.load(path_kmeans)
        print("Da nap thanh cong K-Means model")

        if hasattr(kmeans_model, "n_features_in_"):
            print(f"K-Means yeu cau feature vector {kmeans_model.n_features_in_} chieu")

    except Exception as e:
        print(f"Loi nap K-Means model: {e}")
else:
    print(f"Khong tim thay K-Means model tai {path_kmeans}")

# Nạp model Phase 2
if tf is not None and os.path.exists(path_p2):
    try:
        model_p2 = tf.keras.models.load_model(path_p2)
        print("Da nap thanh cong mo hinh Phase 2: Augmented")
    except Exception as e:
        print(f"Loi nap mo hinh Phase 2: {e}")
else:
    print("Bo qua nap mo hinh Phase 2 (thieu file model hoac tensorflow)")

# Nạp model Augmentation Tool
if tf is not None and get_augmentation_model is not None:
    try:
        aug_tool = get_augmentation_model()
        print("Da nap thanh cong mo hinh Augmentation Tool")
    except Exception as e:
        print(f"Loi nap mo hinh Augmentation Tool: {e}")
else:
    print("Bo qua nap Augmentation Tool (thieu module hoac tensorflow)")

# Nạp model PyTorch Phase 3
if os.path.exists(path_p3):
    try:
        model_p3 = models.resnet18(num_classes=10)
        model_p3.load_state_dict(torch.load(path_p3, map_location=torch.device('cpu')))
        model_p3.eval()
        print("Da nap thanh cong mo hinh Phase 3: CutMix")
    except Exception as e:
        print(f"Loi nap mo hinh Phase 3: {e}")
else:
    print(f"Bo qua nap mo hinh Phase 3 (khong tim thay file tai {path_p3})")

# Nạp model PyTorch Mosaic
if os.path.exists(path_mosaic):
    try:
        model_mosaic = models.resnet18(num_classes=10)
        model_mosaic.load_state_dict(torch.load(path_mosaic, map_location=torch.device('cpu')))
        model_mosaic.eval()
        print("Da nap thanh cong mo hinh Phase 4: Mosaic")
    except Exception as e:
        print(f"Loi nap mo hinh Phase 4: {e}")
else:
    print(f"Bo qua nap mo hinh Phase 4 (khong tim thay file tai {path_mosaic})")


custom_theme = gr.themes.Default(
    primary_hue="blue",
    neutral_hue="gray"
).set(
    # Nền của trang và block
    body_background_fill="#F9FAFB",
    block_background_fill="#FFFFFF",
    block_border_color="#E5E7EB",
    block_border_width="1px",
    # Block label (nhãn góc trên component)
    block_label_background_fill="transparent",
    block_label_text_color="#475569",
    block_label_border_color="transparent",
    block_label_border_width="0px",
    # Button Primary: xanh + chữ trắng
    button_primary_background_fill="#2563EB",
    button_primary_background_fill_hover="#1d4ed8",
    button_primary_text_color="#FFFFFF",
    button_primary_border_color="#2563EB",
    # Button Secondary (bao gồm radio options chưa chọn): xám nhạt + chữ tối
    button_secondary_background_fill="#F1F5F9",
    button_secondary_background_fill_hover="#E2E8F0",
    button_secondary_text_color="#0f172a",
    button_secondary_border_color="#E2E8F0",
    # Input (textbox, textarea): nền trắng + chữ tối
    input_background_fill="#FFFFFF",
    input_border_color="#E5E7EB",
    input_placeholder_color="#94a3b8",
    # Background chung
    background_fill_primary="#FFFFFF",
    background_fill_secondary="#F8FAFC",
)


custom_css = """
/* ============================================================
   FIX CHÍNH: Scope CSS variables vào .gradio-container
   .gradio-container có specificity cao hơn :root
   nên LUÔN thắng theme dù theme load sau
   ============================================================ */
:root {
    color-scheme: light;
}

/* .gradio-container có specificity cao hơn :root nên luôn thắng theme */
.gradio-container {
    --block-background-fill: #ffffff;
    --input-background-fill: #ffffff;
    --body-background-fill: #f8fafc;
    --background-fill-primary: #ffffff;
    --background-fill-secondary: #f8fafc;
    --panel-background-fill: #ffffff;
    --body-text-color: #0f172a;
    --body-text-color-subdued: #0f172a;
    --input-text-color: #0f172a;
    --block-label-text-color: #0f172a;
    --block-label-background-fill: transparent;
    --block-label-border-color: transparent;
    --block-title-text-color: #0f172a;
    --prose-text-color: #0f172a;
    --button-secondary-background-fill: #f1f5f9;
    --button-secondary-background-fill-hover: #e2e8f0;
    --button-secondary-text-color: #0f172a;
    --button-secondary-border-color: #e2e8f0;
    --button-primary-background-fill: #2563eb;
    --button-primary-background-fill-hover: #1d4ed8;
    --button-primary-text-color: #ffffff;
    --border-color-primary: #e5e7eb;
    --color-accent: #2563eb;
    --color-accent-soft: #eff6ff;
    --neutral-50: #f8fafc;
    --neutral-100: #f1f5f9;
    --neutral-200: #e2e8f0;
    --neutral-300: #cbd5e1;
    --neutral-400: #94a3b8;
    --neutral-500: #64748b;
    --neutral-600: #334155;
    --neutral-700: #1e293b;
    --neutral-800: #0f172a;
    --neutral-900: #020617;
}

/* Font hiện đại cho toàn app */
* {
    font-family: "Inter", "Segoe UI", "Roboto", Arial, sans-serif !important;
}

/* Ẩn thanh tab ngang mặc định của Gradio */
div[role="tablist"], .tab-nav {
    display: none !important;
}

/* Ẩn header/footer mặc định Gradio */
header, footer {
    display: none !important;
}
.show-api { display: none !important; }
.theme-selector { display: none !important; }
.icon-buttons, button.icon-button, .menu-button, .select-wrap {
    display: none !important;
}

/* Nền trang chính */
body, .gradio-container {
    background-color: #f8fafc !important;
}

/* Màu chữ mặc định: tối trên nền sáng */
body, .gradio-container {
    color: #0f172a !important;
}

/* BACKUP BROAD RULE: Bắt mọi text thường sang đen
   (dùng khi CSS variable không được apply đúng) */
.gradio-container span:not([class*="file-size"]):not([class*="svelte-"] button *),
.gradio-container p,
.gradio-container label,
.gradio-container [data-testid="block-info"],
.gradio-container .block-label,
.gradio-container .label-wrap span,
.gradio-container .form > span,
.gradio-container fieldset > span,
.gradio-container td,
.gradio-container th {
    color: #0f172a !important;
}

/* Ngoại lệ: button primary/selected giữ chữ trắng */
.gradio-container button.primary:not(#sidebar-menu button),
.gradio-container button.primary:not(#sidebar-menu button) span,
.gradio-container button.primary:not(#sidebar-menu button) *,
.gradio-container button.selected,
.gradio-container button.selected span,
.gradio-container button.selected * {
    color: #ffffff !important;
}

/* =================== SIDEBAR =================== */
.sidebar-panel {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5E7EB !important;
    padding: 20px 14px !important;
    min-height: 100vh !important;
    display: flex !important;
    flex-direction: column !important;
}

#sidebar-menu button.secondary {
    text-align: left !important;
    justify-content: flex-start !important;
    border: none !important;
    background: transparent !important;
    color: #0f172a !important;
    padding: 10px 14px !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    margin-bottom: 4px !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
    font-size: 15px !important;
}
#sidebar-menu button.secondary:hover {
    background-color: #F3F4F6 !important;
    color: #0f172a !important;
}
#sidebar-menu button.secondary * { color: #0f172a !important; }
#sidebar-menu button.secondary:hover * { color: #0f172a !important; }

#sidebar-menu button.primary {
    text-align: left !important;
    justify-content: flex-start !important;
    border: none !important;
    background-color: #EFF6FF !important;
    color: #1e40af !important;
    padding: 10px 14px !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border-left: 4px solid #2563eb !important;
    border-top-left-radius: 0px !important;
    border-bottom-left-radius: 0px !important;
    margin-bottom: 4px !important;
    width: 100% !important;
    box-shadow: none !important;
    font-size: 15px !important;
}
#sidebar-menu button.primary * { color: #1e40af !important; }

.sidebar-hint-box {
    background-color: #F8FAFC !important;
    border: 1px solid #E5E7EB !important;
    padding: 14px !important;
    border-radius: 12px !important;
    color: #0f172a !important;
    font-size: 0.85rem !important;
    margin-top: auto !important;
    box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05) !important;
}
.sidebar-hint-box * { color: #0f172a !important; }

/* =================== LAYOUT =================== */
.content-panel {
    background-color: #f8fafc !important;
    padding: 20px !important;
    min-height: 100vh !important;
}

.custom-card, .block, .panel {
    background-color: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 12px rgba(15,23,42,0.04) !important;
    padding: 14px 18px !important;
    margin-bottom: 12px !important;
}

.card-title {
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    margin-bottom: 10px !important;
    border-bottom: 1px solid #F1F5F9 !important;
    padding-bottom: 6px !important;
}

.gap { gap: 8px !important; }

/* =================== GRADIO COMPONENTS - LIGHT THEME =================== */

/* Block label góc trên: trong suốt, chữ đen */
.gradio-container .block-label,
.gradio-container span.block-label,
.gradio-container .block-label span,
.gradio-container .label-wrap,
.gradio-container .label-wrap span {
    background: transparent !important;
    background-color: transparent !important;
    color: #0f172a !important;
}

/* =============================================================
   GRADIO RADIO GROUP - Gradio 4+ renders radio options as <button>
   elements inside a .wrap container inside a <fieldset>
   ============================================================= */

/* Container của radio group */
.gradio-container fieldset,
.gradio-container .gradio-radio,
.gradio-container [class*="radio"] {
    background-color: #ffffff !important;
    border-color: #e5e7eb !important;
}

/* CÁC BUTTON CHỌN (pill/chip) - CHƯA ĐƯỢC CHỌN: nền xám nhạt, chữ tối */
.gradio-container fieldset .wrap button,
.gradio-container .gradio-radio .wrap button,
.gradio-container fieldset > div button,
.gradio-container [class*="radio"] button {
    background-color: #f1f5f9 !important;
    background: #f1f5f9 !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
}
.gradio-container fieldset .wrap button *,
.gradio-container .gradio-radio .wrap button *,
.gradio-container fieldset > div button *,
.gradio-container [class*="radio"] button * {
    color: #0f172a !important;
}

/* BUTTON ĐÃ ĐƯỢC CHỌN: nền xanh, chữ trắng */
.gradio-container fieldset .wrap button.selected,
.gradio-container .gradio-radio .wrap button.selected,
.gradio-container fieldset > div button.selected,
.gradio-container [class*="radio"] button.selected {
    background-color: #2563eb !important;
    background: #2563eb !important;
    color: #ffffff !important;
    border-color: #2563eb !important;
}
.gradio-container fieldset .wrap button.selected *,
.gradio-container .gradio-radio .wrap button.selected *,
.gradio-container fieldset > div button.selected *,
.gradio-container [class*="radio"] button.selected * {
    color: #ffffff !important;
}

/* Checkbox group: nền trắng, chữ tối */
.gradio-container .checkbox,
.gradio-container [class*="checkbox"],
.gradio-container .gradio-checkboxgroup {
    background-color: #ffffff !important;
    background: #ffffff !important;
    border-color: #e5e7eb !important;
}
.gradio-container .checkbox label,
.gradio-container [class*="checkbox"] label,
.gradio-container .checkbox span,
.gradio-container [class*="checkbox"] span,
.gradio-container .checkbox p,
.gradio-container [class*="checkbox"] p {
    color: #0f172a !important;
    background-color: transparent !important;
}

/* Input radio/checkbox nguyên tố */
input[type="radio"], input[type="checkbox"] {
    border-color: #CBD5E1 !important;
}
input[type="radio"]:checked, input[type="checkbox"]:checked {
    background-color: #2563eb !important;
    border-color: #2563eb !important;
}
.gradio-slider input[type="range"] {
    accent-color: #2563eb !important;
}

/* Textbox, textarea: nền trắng, chữ tối */
.gradio-container .gradio-textbox,
.gradio-container .gradio-textbox input,
.gradio-container .gradio-textbox textarea,
.gradio-container div[class*="textbox"],
.gradio-container div[class*="textbox"] input,
.gradio-container div[class*="textbox"] textarea {
    background-color: #ffffff !important;
    background: #ffffff !important;
    color: #0f172a !important;
    border-color: #cbd5e1 !important;
}
.gradio-container .gradio-textbox *,
.gradio-container div[class*="textbox"] * {
    color: #0f172a !important;
}

/* gr.Label output: nền trắng, chữ tối */
.gradio-container .gradio-label,
.gradio-container div[class*="output"],
.gradio-container div[class*="label"] {
    background-color: #ffffff !important;
    background: #ffffff !important;
    color: #0f172a !important;
    border-color: #cbd5e1 !important;
}
.gradio-container .gradio-label *,
.gradio-container div[class*="output"] *,
.gradio-container div[class*="label"] * {
    color: #0f172a !important;
}

/* Dropdown */
.gradio-textbox input, .gradio-textbox textarea,
.gradio-dropdown select, .gradio-dropdown input {
    background-color: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 6px !important;
    color: #0f172a !important;
}

/* Upload area - CHỈ áp dụng cho upload/file preview, KHÔNG cho .wrap của radio */
.upload-container,
div[data-testid="image"] .wrap,
div[data-testid="file-upload"] .wrap,
.file-preview,
.preview {
    background-color: #F8FAFC !important;
    border: 2px dashed #D1D5DB !important;
    border-radius: 10px !important;
}

/* Dataframe */
.gradio-dataframe table, .gradio-container .gradio-dataframe table {
    background-color: #FFFFFF !important;
    border-collapse: collapse !important;
}
.gradio-dataframe th, .gradio-container .gradio-dataframe th {
    background-color: #F8FAFC !important;
    color: #0f172a !important;
    font-weight: 600 !important;
}
.gradio-dataframe td, .gradio-dataframe td span,
.gradio-container .gradio-dataframe td,
.gradio-container .gradio-dataframe td span {
    background-color: #FFFFFF !important;
    color: #334155 !important;
}

/* =================== BUTTONS =================== */

/* Primary button (ngoài sidebar): xanh -> chữ TRẮNG */
button.primary:not(#sidebar-menu button), .primary-button {
    background-color: #2563eb !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 8px 14px !important;
    transition: background-color 0.2s ease !important;
}
button.primary:not(#sidebar-menu button):hover { background-color: #1d4ed8 !important; }
button.primary:not(#sidebar-menu button) *, .primary-button * { color: #FFFFFF !important; }

/* Secondary button (ngoài sidebar): xám nhạt -> chữ tối */
button.secondary:not(#sidebar-menu button) {
    background-color: #F1F5F9 !important;
    color: #0f172a !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 8px 14px !important;
    transition: background-color 0.2s ease !important;
}
button.secondary:not(#sidebar-menu button):hover { background-color: #E2E8F0 !important; }
button.secondary:not(#sidebar-menu button) * { color: #0f172a !important; }

/* =================== FILE LIST =================== */
.gradio-file,
.file-preview,
.file-preview-row,
.file-row,
.file,
.download-file,
.file-name,
.file-info,
.file-preview ul,
.file-preview li,
div[class*="file-preview"],
div[class*="file-row"],
div[class*="download-file"],
div[class*="file-item"] {
    background-color: #ffffff !important;
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    color: #0f172a !important;
}
.gradio-file *,
.file-preview *,
.file-preview-row *,
.file-row *,
.file *,
.download-file *,
.file-name *,
.file-info *,
.file-preview ul *,
.file-preview li *,
div[class*="file-preview"] *,
div[class*="file-row"] *,
div[class*="download-file"] *,
div[class*="file-item"] * {
    background-color: transparent !important;
    color: #0f172a !important;
}
.file-size, .file-size *, div[class*="file-size"], div[class*="file-size"] * {
    color: #2563eb !important;
}

/* =================== COMPACT HEIGHTS =================== */
.compact-file-uploader {
    max-height: 110px !important;
    min-height: 90px !important;
}
.compact-file-uploader .wrap, .compact-file-uploader .file-preview {
    max-height: 90px !important;
}
.compact-file-downloader {
    max-height: 75px !important;
    min-height: 60px !important;
}
.compact-file-downloader .wrap, .compact-file-downloader .file-preview {
    max-height: 60px !important;
}
.compact-image-preview, .compact-image-preview img, .compact-image-preview .image-container {
    max-height: 160px !important;
    min-height: 120px !important;
    object-fit: contain !important;
}
.compact-gallery, .compact-gallery .grid-wrap {
    max-height: 220px !important;
    overflow-y: auto !important;
}

/* Placeholder */
.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
    color: #94a3b8 !important;
}
"""

with gr.Blocks(title="Food Classification System") as app:
    with gr.Row(equal_height=False):
        # Sidebar cột trái
        with gr.Column(scale=1, min_width=290, elem_classes="sidebar-panel"):
            # Logo / Header
            gr.HTML("""
            <div style="margin-bottom: 24px; text-align: center; border-bottom: 1px solid #E5E7EB; padding-bottom: 16px;">
                <div style="font-size: 2.2rem; margin-bottom: 4px; color: #000000;">🍔</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #000000; line-height: 1.2;">Hệ thống Phân loại</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #000000; line-height: 1.2; margin-top: 4px; margin-bottom: 6px;">Thực phẩm & Tăng cường</div>
                <div style="font-size: 0.725rem; color: #000000; font-family: 'Segoe UI', Arial, sans-serif; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;">Food System Dashboard</div>
            </div>
            """)
            
            # Menu điều hướng
            with gr.Column(elem_id="sidebar-menu"):
                btn_p1 = gr.Button("📊 Phase 1: Baseline", variant="primary")
                btn_p2 = gr.Button("📈 Phase 2: Augmented", variant="secondary")
                btn_p3 = gr.Button("✂️ Phase 3: CutMix", variant="secondary")
                btn_p4 = gr.Button("🧱 Phase 4: Mosaic", variant="secondary")
                btn_tool = gr.Button("🛠️ Data Augmentation Tool", variant="secondary")
                btn_final = gr.Button("🎯 Final Application", variant="secondary")
                btn_batch_inf = gr.Button("🗂️ Batch Inference", variant="secondary")
                btn_batch_aug = gr.Button("⚙️ Trạm Tăng Cường Dữ Liệu", variant="secondary")
            
            # Hộp gợi ý ở cuối Sidebar
            gr.HTML("""
            <div class="sidebar-hint-box" style="margin-top: 30px;">
                <strong>💡 Gợi ý:</strong><br>
                Tăng cường dữ liệu giúp mô hình học đa dạng hơn, cải thiện khả năng khái quát và độ chính xác của mạng CNN.
            </div>
            """)
            
        # Nội dung chính cột phải
        with gr.Column(scale=4, elem_classes="content-panel"):
            with gr.Tabs() as tabs:
                with gr.Tab("Phase 1: Baseline", id=0):
                    gr.HTML("""
                    <div style="background-color: #FFFFFF; border: 1px solid #E5E7EB; padding: 10px 16px; border-radius: 12px; margin-bottom: 12px; box-shadow: 0 1px 3px 0 rgba(0,0,0,0.05);">
                        <h2 style="font-size: 1.2rem; font-weight: 800; color: #0f172a; margin: 0 0 2px 0;">📊 Phase 1: Baseline Model</h2>
                        <p style="font-size: 0.825rem; color: #475569; margin: 0;">Mô hình CNN Baseline thuần túy được huấn luyện trên tập dữ liệu gốc 10 lớp món ăn để làm mốc đánh giá so sánh.</p>
                    </div>
                    """)
                    render_tab_phase1(model_p1)

                with gr.Tab("Phase 2: Augmented", id=1):
                    gr.HTML("""
                    <div style="background-color: #FFFFFF; border: 1px solid #E5E7EB; padding: 10px 16px; border-radius: 12px; margin-bottom: 12px; box-shadow: 0 1px 3px 0 rgba(0,0,0,0.05);">
                        <h2 style="font-size: 1.2rem; font-weight: 800; color: #0f172a; margin: 0 0 2px 0;">📈 Phase 2: Augmented Model</h2>
                        <p style="font-size: 0.825rem; color: #475569; margin: 0;">Mô hình CNN được huấn luyện kết hợp các kỹ thuật tăng cường dữ liệu On-the-fly cơ bản nhằm cải thiện khả năng tổng quát hóa.</p>
                    </div>
                    """)
                    render_tab_phase2(model_p2)

                with gr.Tab("Phase 3: CutMix", id=2):
                    gr.HTML("""
                    <div style="background-color: #FFFFFF; border: 1px solid #E5E7EB; padding: 10px 16px; border-radius: 12px; margin-bottom: 12px; box-shadow: 0 1px 3px 0 rgba(0,0,0,0.05);">
                        <h2 style="font-size: 1.2rem; font-weight: 800; color: #0f172a; margin: 0 0 2px 0;">✂️ Phase 3: CutMix Model</h2>
                        <p style="font-size: 0.825rem; color: #475569; margin: 0;">Mô hình PyTorch ResNet18 được tinh chỉnh với chiến lược CutMix - cắt và dán các vùng ảnh ngẫu nhiên.</p>
                    </div>
                    """)
                    render_tab_phase3(model_p3)

                with gr.Tab("Phase 4: Mosaic", id=3):
                    gr.HTML("""
                    <div style="background-color: #FFFFFF; border: 1px solid #E5E7EB; padding: 10px 16px; border-radius: 12px; margin-bottom: 12px; box-shadow: 0 1px 3px 0 rgba(0,0,0,0.05);">
                        <h2 style="font-size: 1.2rem; font-weight: 800; color: #0f172a; margin: 0 0 2px 0;">🧱 Phase 4: Mosaic Model</h2>
                        <p style="font-size: 0.825rem; color: #475569; margin: 0;">Mô hình PyTorch ResNet18 sử dụng kỹ thuật ghép ảnh Mosaic (4 góc) để tăng cường bối cảnh vật thể.</p>
                    </div>
                    """)
                    render_tab_mosaic(model_mosaic)

                with gr.Tab("Data Augmentation Tool", id=4):
                    gr.HTML("""
                    <div style="background-color: #FFFFFF; border: 1px solid #E5E7EB; padding: 10px 16px; border-radius: 12px; margin-bottom: 12px; box-shadow: 0 1px 3px 0 rgba(0,0,0,0.05);">
                        <h2 style="font-size: 1.2rem; font-weight: 800; color: #0f172a; margin: 0 0 2px 0;">🛠️ Data Augmentation Tool</h2>
                        <p style="font-size: 0.825rem; color: #475569; margin: 0;">Công cụ trực quan hóa thử nghiệm nhanh các phép biến dạng ảnh đơn lẻ trước khi xuất bản hàng loạt.</p>
                    </div>
                    """)
                    render_tab_augmentation(aug_tool)

                with gr.Tab("Final Application", id=5):
                    gr.HTML("""
                    <div style="background-color: #FFFFFF; border: 1px solid #E5E7EB; padding: 10px 16px; border-radius: 12px; margin-bottom: 12px; box-shadow: 0 1px 3px 0 rgba(0,0,0,0.05);">
                        <h2 style="font-size: 1.2rem; font-weight: 800; color: #0f172a; margin: 0 0 2px 0;">🎯 Final Application</h2>
                        <p style="font-size: 0.825rem; color: #475569; margin: 0;">Ứng dụng tích hợp đầu cuối: Chọn mô hình phân loại và hiển thị chi tiết phân tích phân cụm hỗ trợ từ K-Means.</p>
                    </div>
                    """)
                    render_tab_final_app(
                        model_p1,
                        model_p2,
                        model_p3,
                        model_mosaic,
                        kmeans_model,
                        feature_extractor
                    )

                with gr.Tab("Batch Inference", id=6):
                    gr.HTML("""
                    <div style="background-color: #FFFFFF; border: 1px solid #E5E7EB; padding: 10px 16px; border-radius: 12px; margin-bottom: 12px; box-shadow: 0 1px 3px 0 rgba(0,0,0,0.05);">
                        <h2 style="font-size: 1.2rem; font-weight: 800; color: #0f172a; margin: 0 0 2px 0;">🗂️ Batch Inference</h2>
                        <p style="font-size: 0.825rem; color: #475569; margin: 0;">Hỗ trợ phân loại hàng loạt danh sách ảnh, kết xuất báo cáo và đóng gói tệp ZIP chia thư mục tự động theo nhãn dự đoán.</p>
                    </div>
                    """)
                    render_tab_batch_inference(
                        model_p1,
                        model_p2,
                        model_p3,
                        model_mosaic,
                        kmeans_model,
                        feature_extractor
                    )

                with gr.Tab("Trạm Tăng Cường Dữ Liệu", id=7):
                    render_tab_batch_augmentation()
                    
    # Cơ chế điều hướng
    menu_buttons = [btn_p1, btn_p2, btn_p3, btn_p4, btn_tool, btn_final, btn_batch_inf, btn_batch_aug]
    
    def make_navigate(tab_id):
        def navigate():
            updates = [gr.Tabs(selected=tab_id)]
            for i in range(8):
                updates.append(gr.Button(variant="primary" if i == tab_id else "secondary"))
            return updates
        return navigate

    for tab_idx, btn in enumerate(menu_buttons):
        btn.click(
            fn=make_navigate(tab_idx),
            inputs=None,
            outputs=[tabs] + menu_buttons
        )

    # Cơ chế điều hướng
    menu_buttons = [btn_p1, btn_p2, btn_p3, btn_p4, btn_tool, btn_final, btn_batch_inf, btn_batch_aug]
    
    def make_navigate(tab_id):
        def navigate():
            updates = [gr.Tabs(selected=tab_id)]
            for i in range(8):
                updates.append(gr.Button(variant="primary" if i == tab_id else "secondary"))
            return updates
        return navigate

    for tab_idx, btn in enumerate(menu_buttons):
        btn.click(
            fn=make_navigate(tab_idx),
            inputs=None,
            outputs=[tabs] + menu_buttons
        )

if __name__ == "__main__":
    app.launch(theme=custom_theme, css=custom_css)