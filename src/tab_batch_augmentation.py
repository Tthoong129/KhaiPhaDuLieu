import os
import zipfile
import tempfile
import random
import numpy as np
import csv
import time
from PIL import Image, ImageFilter
import torchvision.transforms as T
import gradio as gr
from aug_cutmix import get_cutmix_bbox
from aug_mosaic import pil_mosaic

def apply_single_augmentation(img, option, bg_images):
    """Áp dụng 1 hiệu ứng duy nhất (Dành cho Demo)"""
    if option == "Flip — Ngang / Dọc":
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        
    elif option == "Rotate — 0°–360°":
        # Xoay nhẹ an toàn cho training (xoay góc ngẫu nhiên -15 đến 15 độ)
        angle = random.uniform(-15, 15)
        return img.rotate(angle, resample=Image.BICUBIC, fillcolor=(127,127,127))
        
    elif option == "Zoom — 80%–120%":
        # Phóng to/Thu nhỏ nhẹ (tỷ lệ 0.9 đến 1.1)
        zoom_factor = random.uniform(0.9, 1.1)
        w, h = img.size
        if zoom_factor > 1.0:
            nw, nh = int(w / zoom_factor), int(h / zoom_factor)
            left = (w - nw) // 2
            top = (h - nh) // 2
            return img.crop((left, top, left + nw, top + nh)).resize((w, h), Image.Resampling.LANCZOS)
        else:
            nw, nh = int(w * zoom_factor), int(h * zoom_factor)
            resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
            new_img = Image.new("RGB", (w, h), (127, 127, 127))
            left = (w - nw) // 2
            top = (h - nh) // 2
            new_img.paste(resized, (left, top))
            return new_img

    elif option == "Color Jitter — Màu sắc / Độ sáng":
        jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        return jitter(img)
        
    elif option == "Blur — Gaussian Blur":
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 2.0)))
        
    elif option == "CutMix — Trộn ảnh ngẫu nhiên":
        if bg_images:
            bg_img = random.choice(bg_images).resize(img.size, Image.Resampling.LANCZOS)
            lam = np.random.beta(1.0, 1.0)
            bbx1, bby1, bbx2, bby2 = get_cutmix_bbox(img.size, lam)
            if bbx2 > bbx1 and bby2 > bby1:
                res_img = img.copy()
                region = bg_img.crop((bbx1, bby1, bbx2, bby2))
                res_img.paste(region, (bbx1, bby1))
                return res_img
        return img
            
    elif option == "Mosaic — Ghép ảnh dạng lưới":
        if bg_images:
            return pil_mosaic(img, bg_images, size=img.size)
        return img
        
    return img


def apply_pipeline_augmentation(img, options, bg_images):
    """Trộn ngẫu nhiên các hiệu ứng an toàn (Dành cho Training)"""
    if "Flip — Ngang / Dọc" in options and random.random() < 0.5:
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        
    if "Rotate — 0°–360°" in options:
        angle = random.uniform(-15, 15)
        img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=(127,127,127))
        
    if "Zoom — 80%–120%" in options:
        zoom_factor = random.uniform(0.9, 1.1)
        w, h = img.size
        if zoom_factor > 1.0:
            nw, nh = int(w / zoom_factor), int(h / zoom_factor)
            left = (w - nw) // 2
            top = (h - nh) // 2
            img = img.crop((left, top, left + nw, top + nh)).resize((w, h), Image.Resampling.LANCZOS)
        else:
            nw, nh = int(w * zoom_factor), int(h * zoom_factor)
            resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
            new_img = Image.new("RGB", (w, h), (127, 127, 127))
            left = (w - nw) // 2
            top = (h - nh) // 2
            new_img.paste(resized, (left, top))
            img = new_img

    if "Color Jitter — Màu sắc / Độ sáng" in options:
        jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        img = jitter(img)
        
    if "Blur — Gaussian Blur" in options and random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        
    if "CutMix — Trộn ảnh ngẫu nhiên" in options and bg_images and random.random() < 0.3:
        bg_img = random.choice(bg_images).resize(img.size, Image.Resampling.LANCZOS)
        lam = np.random.beta(1.0, 1.0)
        bbx1, bby1, bbx2, bby2 = get_cutmix_bbox(img.size, lam)
        if bbx2 > bbx1 and bby2 > bby1:
            res_img = img.copy()
            region = bg_img.crop((bbx1, bby1, bbx2, bby2))
            res_img.paste(region, (bbx1, bby1))
            img = res_img
            
    if "Mosaic — Ghép ảnh dạng lưới" in options and bg_images and random.random() < 0.3:
        img = pil_mosaic(img, bg_images, size=img.size)
        
    return img


def handle_batch_augmentation(uploaded_files, aug_options, mode, multiplier_val):
    if not uploaded_files:
        return None, "### ⚠️ Lỗi: Chưa tải lên tệp tin dữ liệu gốc nào.", """
        | Thông số | Giá trị |
        | :--- | :--- |
        | **Trạng thái** | Lỗi |
        | **Số lượng ảnh gốc** | 0 ảnh |
        | **Tổng số ảnh sau tăng cường** | 0 ảnh |
        | **Thời gian tạo** | -- |
        """
        
    if not aug_options:
        return None, "### ⚠️ Lỗi: Vui lòng chọn ít nhất một kỹ thuật tăng cường.", """
        | Thông số | Giá trị |
        | :--- | :--- |
        | **Trạng thái** | Lỗi |
        | **Số lượng ảnh gốc** | 0 ảnh |
        | **Tổng số ảnh sau tăng cường** | 0 ảnh |
        | **Thời gian tạo** | -- |
        """
        
    start_time = time.time()
    temp_dir = tempfile.mkdtemp()
    input_extract_dir = os.path.join(temp_dir, "extracted")
    output_dataset_dir = os.path.join(temp_dir, "Augmented_Dataset")
    os.makedirs(input_extract_dir, exist_ok=True)
    os.makedirs(output_dataset_dir, exist_ok=True)
    
    raw_images = []
    warnings_list = []
    
    # 1. Thu thập ảnh đầu vào
    files_list = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
    
    has_zip = False
    for f_obj in files_list:
        f_path = f_obj if isinstance(f_obj, str) else f_obj.name
        if f_path.lower().endswith(".zip"):
            has_zip = True
            try:
                with zipfile.ZipFile(f_path, 'r') as zip_ref:
                    zip_ref.extractall(input_extract_dir)
            except Exception as e:
                warnings_list.append(f"Không thể giải nén tệp zip '{os.path.basename(f_path)}': {str(e)}")
                
    if has_zip:
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        for root, dirs, files in os.walk(input_extract_dir):
            for file in files:
                if file.lower().endswith(valid_exts):
                    abs_path = os.path.join(root, file)
                    rel_dir = os.path.basename(root)
                    if rel_dir == "extracted" or not rel_dir:
                        class_name = "Unclassified"
                    else:
                        class_name = rel_dir
                    raw_images.append((abs_path, class_name, file))
    else:
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        for f_obj in files_list:
            f_path = f_obj if isinstance(f_obj, str) else f_obj.name
            fname = os.path.basename(f_path)
            if fname.lower().endswith(valid_exts):
                raw_images.append((f_path, "Unclassified", fname))
                
    if not raw_images:
        return None, "### ⚠️ Lỗi: Không tìm thấy hình ảnh hợp lệ nào trong tệp tải lên (Chỉ chấp nhận .jpg, .jpeg, .png, .bmp, .webp).", """
        | Thông số | Giá trị |
        | :--- | :--- |
        | **Trạng thái** | Không tìm thấy ảnh |
        | **Số lượng ảnh gốc** | 0 ảnh |
        | **Tổng số ảnh sau tăng cường** | 0 ảnh |
        | **Thời gian tạo** | -- |
        """
        
    orig_total_count = len(raw_images)
    
    # Ràng buộc 100 ảnh gốc
    if orig_total_count > 100:
        warnings_list.append(f"Dataset tải lên có {orig_total_count} ảnh gốc, vượt quá giới hạn cho phép (100 ảnh). Hệ thống chỉ xử lý 100 ảnh đầu tiên để duy trì hiệu năng ổn định.")
        raw_images = raw_images[:100]
        
    bg_images = []
    for img_info in raw_images:
        try:
            bg_images.append(Image.open(img_info[0]).convert("RGB"))
        except:
            pass
            
    is_training_mode = "Trộn" in mode
    report_data = []
    
    augmented_count = 0
    max_outputs_reached = False
    
    for idx, (img_path, class_name, orig_filename) in enumerate(raw_images):
        if max_outputs_reached:
            break
            
        name_we, ext = os.path.splitext(orig_filename)
        if not ext:
            ext = ".jpg"
            
        try:
            class_out_dir = os.path.join(output_dataset_dir, class_name)
            os.makedirs(class_out_dir, exist_ok=True)
            
            img = Image.open(img_path).convert("RGB")
            
            orig_out_name = f"{name_we}_orig{ext}"
            img.save(os.path.join(class_out_dir, orig_out_name))
            report_data.append([class_name, orig_filename, orig_out_name, "Original", "Single"])
            augmented_count += 1
            
            if is_training_mode:
                for c_idx in range(int(multiplier_val)):
                    if augmented_count >= 1000:
                        max_outputs_reached = True
                        warnings_list.append("Đã đạt giới hạn tối đa 1000 ảnh đầu ra. Quá trình tăng cường dừng lại tại đây.")
                        break
                        
                    aug_img = apply_pipeline_augmentation(img.copy(), aug_options, bg_images)
                    aug_out_name = f"{name_we}_aug_stack_{c_idx+1}{ext}"
                    aug_img.save(os.path.join(class_out_dir, aug_out_name))
                    report_data.append([class_name, orig_filename, aug_out_name, "Mixed Pipeline", "Pipeline Stacking"])
                    augmented_count += 1
            else:
                suffix_map = {
                    "Flip — Ngang / Dọc": "flip",
                    "Rotate — 0°–360°": "rotate",
                    "Zoom — 80%–120%": "zoom",
                    "Color Jitter — Màu sắc / Độ sáng": "color",
                    "Blur — Gaussian Blur": "blur",
                    "CutMix — Trộn ảnh ngẫu nhiên": "cutmix",
                    "Mosaic — Ghép ảnh dạng lưới": "mosaic"
                }
                
                for option in aug_options:
                    if augmented_count >= 1000:
                        max_outputs_reached = True
                        warnings_list.append("Đã đạt giới hạn tối đa 1000 ảnh đầu ra. Quá trình tăng cường dừng lại tại đây.")
                        break
                        
                    aug_img = apply_single_augmentation(img.copy(), option, bg_images)
                    suffix = suffix_map.get(option, 'aug')
                    aug_out_name = f"{name_we}_aug_{suffix}{ext}"
                    aug_img.save(os.path.join(class_out_dir, aug_out_name))
                    report_data.append([class_name, orig_filename, aug_out_name, option, "Single Effect"])
                    augmented_count += 1
                    
        except Exception as e:
            warnings_list.append(f"Lỗi xử lý tệp '{orig_filename}': {str(e)}")
            
    # Tạo CSV report
    report_csv_path = os.path.join(output_dataset_dir, "augmentation_report.csv")
    try:
        with open(report_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["class", "original_file", "augmented_file", "technique", "mode"])
            writer.writerows(report_data)
    except Exception as e:
        warnings_list.append(f"Không thể tạo file báo cáo CSV: {str(e)}")
        
    # Nén zip đầu ra
    zip_output_path = os.path.join(temp_dir, "Augmented_Dataset.zip")
    try:
        with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_dataset_dir):
                for file in files:
                    abs_filepath = os.path.join(root, file)
                    rel_filepath = os.path.relpath(abs_filepath, output_dataset_dir)
                    zipf.write(abs_filepath, rel_filepath)
    except Exception as e:
        return None, f"### ⚠️ Lỗi khi tạo file nén đầu ra: {str(e)}", ""
        
    duration = time.time() - start_time
    
    status_md = "### 🎉 Tăng cường dữ liệu hoàn tất thành công!\n\n"
    if warnings_list:
        status_md += "**⚠️ Cảnh báo và giới hạn:**\n"
        for warn in warnings_list:
            status_md += f"- {warn}\n"
        status_md += "\n"
    else:
        status_md += "Không có lỗi hay cảnh báo nào phát sinh trong quá trình thực hiện.\n\n"
        
    info_md = f"""
| Thông số | Giá trị |
| :--- | :--- |
| **Trạng thái** | {"Thành công (Có cảnh báo/Cắt giảm)" if warnings_list else "Hoàn thành xuất sắc"} |
| **Số lượng ảnh gốc đã xử lý** | {len(raw_images)} / {orig_total_count} ảnh |
| **Tổng số ảnh trong ZIP đầu ra** | {augmented_count} ảnh |
| **Thời gian xử lý** | {duration:.2f} giây |
"""
    
    return zip_output_path, status_md, info_md


def render_tab_batch_augmentation():
    # Header Hero Card lớn - Gọn gàng hơn
    gr.HTML("""
    <div style="background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%); border: 1px solid #BFDBFE; padding: 18px 24px; border-radius: 16px; display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
        <div style="flex: 1; padding-right: 20px;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 6px;">
                <span style="font-size: 2.2rem;">⚙️</span>
                <h1 style="font-size: 1.5rem; font-weight: 800; color: #1E3A8A; margin: 0;">Trạm Tăng Cường Dữ Liệu</h1>
            </div>
            <p style="font-size: 0.8rem; font-weight: 700; color: #3B82F6; margin: 0 0 6px 0; text-transform: uppercase; letter-spacing: 0.05em;">Data Augmentation Station</p>
            <p style="font-size: 0.9rem; color: #4B5563; margin: 0; line-height: 1.4;">
                Tính năng này giúp bạn nhân bản và làm phong phú dữ liệu (Data Augmentation) để cải thiện hiệu suất mô hình huấn luyện AI. Hỗ trợ tệp tin nén chứa các thư mục nhãn lớp.
            </p>
        </div>
        <div style="font-size: 4rem; user-select: none; opacity: 0.8; padding-right: 16px;">
            📚
        </div>
    </div>
    """)
    
    with gr.Row(equal_height=False):
        # Cột Trái: Upload & Cấu hình (scale=3)
        with gr.Column(scale=3):
            # Khối 1: Tải lên danh sách ảnh
            with gr.Column(elem_classes="custom-card"):
                gr.HTML("<div class='card-title'>1. Tải lên danh sách ảnh gốc</div>")
                file_uploader = gr.File(
                    label="Kéo & thả file ZIP dataset hoặc nhiều ảnh rời vào đây",
                    file_count="multiple",
                    type="filepath",
                    height=100,
                    show_label=False,
                    elem_classes=["compact-file-uploader"]
                )
                gr.HTML("""
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">
                    💡 <strong>Cấu trúc ZIP:</strong> Nén các thư mục lớp (ví dụ: Burger/, Pizza/...). Ảnh tăng cường sinh ra sẽ nằm trong các thư mục lớp tương ứng trong zip kết quả.
                </div>
                """)
                
            # Khối 2: Tùy chọn tăng cường
            with gr.Column(elem_classes="custom-card"):
                gr.HTML("<div class='card-title'>2. Tùy chọn tăng cường dữ liệu</div>")
                
                # Checkbox Group cho kỹ thuật cơ bản
                basic_options = gr.CheckboxGroup(
                    choices=[
                        "Flip — Ngang / Dọc",
                        "Rotate — 0°–360°",
                        "Zoom — 80%–120%",
                        "Color Jitter — Màu sắc / Độ sáng",
                        "Blur — Gaussian Blur"
                    ],
                    value=[
                        "Flip — Ngang / Dọc",
                        "Rotate — 0°–360°",
                        "Zoom — 80%–120%"
                    ],
                    label="Kỹ thuật cơ bản (Safe Augmentations for Training - Khuyên dùng)"
                )
                
                # Checkbox Group cho kỹ thuật nâng cao
                advanced_options = gr.CheckboxGroup(
                    choices=[
                        "CutMix — Trộn ảnh ngẫu nhiên",
                        "Mosaic — Ghép ảnh dạng lưới"
                    ],
                    value=[],
                    label="Kỹ thuật nâng cao / Demo (Cảnh báo: Có thể làm sai lệch nhãn gốc)"
                )
                
                gr.HTML("""
                <div style="padding: 8px 12px; background-color: #FEF3C7; border-left: 4px solid #F59E0B; border-radius: 6px; font-size: 0.825rem; color: #92400E; margin-bottom: 12px;">
                    ⚠️ <strong>Lưu ý:</strong> CutMix và Mosaic trộn ảnh từ các nhãn khác nhau. Tránh chọn khi xuất dataset để train folder-label thông thường.
                </div>
                """)
                
                # Chọn chế độ
                aug_mode = gr.Radio(
                    choices=[
                        "Trộn ngẫu nhiên (Pipeline Stacking - Tốt nhất để Train AI)", 
                        "Tách lẻ từng hiệu ứng (Dễ nhìn - Dành cho Demo)"
                    ],
                    value="Trộn ngẫu nhiên (Pipeline Stacking - Tốt nhất để Train AI)",
                    label="Chế độ sinh dữ liệu"
                )
                
                # Bộ tăng giảm số lần copies bằng nút + -
                gr.HTML("<div style='font-size: 0.85rem; font-weight: 600; color: #374151; margin-bottom: 4px;'>Số lần nhân bản (copies) - Chỉ áp dụng cho chế độ Trộn</div>")
                with gr.Row():
                    btn_minus = gr.Button("➖ Giảm", variant="secondary", scale=1)
                    copies_display = gr.Number(value=5, show_label=False, precision=0, interactive=False, scale=2)
                    btn_plus = gr.Button("➕ Tăng", variant="secondary", scale=1)
                    
                copies_state = gr.State(5)
                
                def decrease_copies(val):
                    new_val = max(1, val - 1)
                    return new_val, new_val
                    
                def increase_copies(val):
                    new_val = min(20, val + 1)
                    return new_val, new_val
                    
                btn_minus.click(fn=decrease_copies, inputs=copies_state, outputs=[copies_state, copies_display])
                btn_plus.click(fn=increase_copies, inputs=copies_state, outputs=[copies_state, copies_display])
                
            # Khối 3: Nút chạy hành động
            with gr.Row():
                clear_btn = gr.Button("Xóa tất cả", variant="secondary", scale=1)
                run_btn = gr.Button("🚀 Tạo dữ liệu tăng cường", variant="primary", scale=2)

        # Cột Phải: Kết quả & Thống kê & Mẹo (scale=2) - Nhỏ gọn và cân đối hơn
        with gr.Column(scale=2):
            # Khối 1: Tải file Zip kết quả & Thông tin kết quả (Gom chung)
            with gr.Column(elem_classes="custom-card"):
                gr.HTML("<div class='card-title'>Kết quả xử lý</div>")
                download_output = gr.File(label="Tải file Zip kết quả", type="filepath", height=70, elem_classes=["compact-file-downloader"])
                status_display = gr.Markdown("*Trạng thái: Chờ tải dữ liệu...*")
                
                gr.HTML("<div style='margin-top: 14px; border-top: 1px solid #F1F5F9; padding-top: 10px;'></div>")
                gr.HTML("<div style='font-size: 0.9rem; font-weight: 700; color: #1E3A8A; margin-bottom: 6px;'>Thống kê chi tiết:</div>")
                info_display = gr.Markdown("""
| Thông số | Giá trị |
| :--- | :--- |
| **Trạng thái** | Chưa bắt đầu |
| **Ảnh gốc đã xử lý** | 0 ảnh |
| **Tổng số ảnh sau tăng cường** | 0 ảnh |
| **Thời gian tạo** | -- |
""")
                
            # Khối 2: Mẹo sử dụng (Làm thấp lại)
            with gr.Column(elem_classes="custom-card"):
                gr.HTML("<div class='card-title'>Mẹo sử dụng</div>")
                gr.HTML("""
                <ul style="padding-left: 16px; font-size: 0.825rem; color: #4B5563; line-height: 1.5; margin: 0;">
                    <li><strong>Giới hạn:</strong> Tải lên tối đa 100 ảnh gốc và xuất tối đa 1000 ảnh để tránh quá tải.</li>
                    <li><strong>Báo cáo CSV:</strong> File report.csv trong file ZIP đầu ra liệt kê rõ nguồn gốc của từng file ảnh tăng cường và kỹ thuật được áp dụng.</li>
                </ul>
                """)
                
    # Gắn sự kiện click chạy
    def process_augmentation(files, basic_opts, adv_opts, mode, copies):
        all_opts = basic_opts + adv_opts
        return handle_batch_augmentation(files, all_opts, mode, copies)
        
    run_btn.click(
        fn=process_augmentation,
        inputs=[file_uploader, basic_options, advanced_options, aug_mode, copies_state],
        outputs=[download_output, status_display, info_display]
    )
    
    def reset_fields():
        return None, None, "*Trạng thái: Đã xóa. Sẵn sàng xử lý tập dữ liệu mới.*", """
| Thông số | Giá trị |
| :--- | :--- |
| **Trạng thái** | Chưa bắt đầu |
| **Ảnh gốc đã xử lý** | 0 ảnh |
| **Tổng số ảnh sau tăng cường** | 0 ảnh |
| **Thời gian tạo** | -- |
""", 5, 5
        
    clear_btn.click(
        fn=reset_fields,
        inputs=None,
        outputs=[file_uploader, download_output, status_display, info_display, copies_state, copies_display]
    )
