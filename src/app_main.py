import os
import gradio as gr

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
import torch
import torchvision.models as models


current_dir = os.path.dirname(os.path.abspath(__file__))
path_p1 = os.path.join(current_dir, '../models/baseline_model_p1.keras')
path_p2 = os.path.join(current_dir, '../models/augmented_model_p2.keras')
path_p3 = os.path.join(current_dir, '../models/AI_FastFood_CutMix1.pth')
path_mosaic = os.path.join(current_dir, '../models/fastfood_resnet18_mosaic.pth')

model_p1 = None
model_p2 = None
aug_tool = None
model_p3 = None
model_mosaic = None

# Nạp model Phase 1
if tf is not None and os.path.exists(path_p1):
    try:
        model_p1 = tf.keras.models.load_model(path_p1)
        print("Da nap thanh cong mo hinh Phase 1: Baseline")
    except Exception as e:
        print(f"Loi nap mo hinh Phase 1: {e}")
else:
    print("Bo qua nap mo hinh Phase 1 (thieu file model hoac tensorflow)")

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
    neutral_hue="slate"
).set(
    body_background_fill="#F9FAFB",         
    block_background_fill="#FFFFFF",          
    block_border_color="#BFDBFE",            
    block_border_width="1px",
    button_primary_background_fill="#0EA5E9", 
    button_primary_background_fill_hover="#0284C7",
    button_primary_text_color="#FFFFFF"       
)


with gr.Blocks(theme=custom_theme, title="Food Classification System") as app:
    gr.Markdown("## Hệ thống Phân loại Thực phẩm & Tăng cường Dữ liệu")
    
    with gr.Tab("Phase 1: Baseline"):
        render_tab_phase1(model_p1)

    with gr.Tab("Phase 2: Augmented"):
        render_tab_phase2(model_p2)

    with gr.Tab("Phase 3: CutMix"):
        render_tab_phase3(model_p3)

    with gr.Tab("Phase 4: Mosaic"):
        render_tab_mosaic(model_mosaic)

    with gr.Tab("Data Augmentation Tool"):
        render_tab_augmentation(aug_tool)

if __name__ == "__main__":
    app.launch()