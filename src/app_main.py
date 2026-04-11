import os
import tensorflow as tf
import gradio as gr

from augmentor import get_augmentation_model
from tab_phase1 import render_tab_phase1
from tab_phase2 import render_tab_phase2
from tab_augmentation import render_tab_augmentation

# --- 1. SETUP ĐƯỜNG DẪN & NẠP MODEL ---
current_dir = os.path.dirname(os.path.abspath(__file__))
path_p1 = os.path.join(current_dir, '../models/baseline_model_p1.keras')
path_p2 = os.path.join(current_dir, '../models/augmented_model_p2.keras')

try:
    model_p1 = tf.keras.models.load_model(path_p1)
    model_p2 = tf.keras.models.load_model(path_p2)
    aug_tool = get_augmentation_model()
except Exception as e:
    print(f"Lỗi hệ thống - Không thể nạp mô hình: {e}")
    model_p1, model_p2, aug_tool = None, None, None

# --- 2. CẤU HÌNH GIAO DIỆN XANH - TRẮNG ---
custom_theme = gr.themes.Default(
    primary_hue="blue",
    neutral_hue="slate"
).set(
    body_background_fill="#F9FAFB",          # Nền trang web trắng xám cực nhẹ
    block_background_fill="#FFFFFF",          # Nền các khung ảnh trắng tinh
    block_border_color="#BFDBFE",             # Viền khung màu xanh nhạt
    block_border_width="1px",
    button_primary_background_fill="#0EA5E9", # Nút bấm màu xanh trời chủ đạo
    button_primary_background_fill_hover="#0284C7",
    button_primary_text_color="#FFFFFF"       # Chữ trên nút màu trắng
)

# --- 3. LIÊN KẾT GIAO DIỆN CHÍNH ---
with gr.Blocks(theme=custom_theme, title="Food Classification System") as app:
    gr.Markdown("## Hệ thống Phân loại Thực phẩm & Tăng cường Dữ liệu")
    
    with gr.Tab("Phase 1: Baseline"):
        render_tab_phase1(model_p1)

    with gr.Tab("Phase 2: Augmented"):
        render_tab_phase2(model_p2)

    with gr.Tab("Data Augmentation Tool"):
        render_tab_augmentation(aug_tool)

if __name__ == "__main__":
    app.launch()