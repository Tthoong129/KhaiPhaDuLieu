# Đồ án: Phân loại và Tăng cường Dữ liệu Ảnh Thực phẩm

**Đề tài 5:** Xây dựng ứng dụng tăng cường dữ liệu ảnh
## Link data set:https://www.kaggle.com/datasets/utkarshsaxenadn/fast-food-classification-dataset
---

## 1. Giới thiệu tổng quan

Repository này chứa mã nguồn và báo cáo cho đồ án môn Khai phá dữ liệu. Mục tiêu của dự án là giải quyết bài toán: **Làm sao để mô hình nhận diện học tốt hơn khi dữ liệu bị hạn chế?** Thay vì chỉ train một mạng nơ-ron nhận diện 10 loại thức ăn nhanh (Fast Food) thông thường, dự án đi sâu vào việc áp dụng kỹ thuật Tăng cường dữ liệu (Data Augmentation) để làm phong phú tập huấn luyện, đồng thời dùng thuật toán Phân cụm (Clustering) để gom nhóm và phân tích độ tương đồng của dữ liệu ảnh.

---

## 2. Nền tảng lý thuyết

Dự án được xây dựng dựa trên 3 khối kiến thức chính:

* **Mạng Nơ-ron Tích chập (CNN):** Đóng vai trò là "bộ não" phân loại. Mạng dùng các màng lọc (Convolution) để tự động rút trích các góc cạnh, màu sắc đặc trưng của món ăn.
* **Tăng cường dữ liệu (Data Augmentation):** Khi số lượng ảnh ít, AI rất dễ bị "học vẹt" (Overfitting). Kỹ thuật này sinh ra các phiên bản ảnh mới (lật ngang, xoay nghiêng, phóng to...) ngay trong lúc train để ép AI học tổng quát hơn.
* **Phân cụm K-Means (Clustering):** Thuật toán gom nhóm không giám sát. K-Means tối thiểu hóa tổng bình phương khoảng cách từ các bức ảnh đến tâm của cụm (Hàm mục tiêu WCSS):

$$J = \sum_{j=1}^{k} \sum_{i=1}^{n_j} ||x_i^{(j)} - c_j||^2$$

  Trong đồ án, K-Means đóng vai trò là "Support Layer". Nó gom các ảnh có chung tone màu hoặc góc chụp vào một cụm để giúp phân tích lý do tại sao mạng CNN lại đoán nhầm.

---

## 3. Quá trình thực hiện (Pipeline)

*(Để kết quả đánh giá khách quan nhất, tất cả mô hình ở các Phase đều được cấu hình chung số lượng Epochs và Batch Size).*

* **Phase 1 (Baseline):** Tiền xử lý tập data Kaggle (resize `150x150`, chia tỷ lệ 400 Train - 100 Valid - 100 Test). Train một mạng CNN thuần túy làm mốc so sánh. Lớp cuối của mạng này được dùng để rút trích Vector đặc trưng.
* **Phase 2 (Clustering & Basic Aug):** Dùng K-Means phân cụm các vector đặc trưng từ Phase 1 thành 10 nhóm. Song song đó, viết một Module Augmentation dùng chung (On-the-fly) với các phép biến đổi cơ bản (Flip, Rotate...) để train một mạng Deep CNN sâu hơn.
* **Phase 3 (Advanced Aug):** Đẩy cấp độ lên cao hơn bằng cách dùng các thuật toán nâng cao như MixUp/CutMix (tiên tiến trong 5 năm gần đây) để tối ưu mô hình.
* **Phase 4 & 5 (App & Analysis):** Tích hợp tất cả vào Web App. Ứng dụng không chỉ dự đoán món ăn mà còn show ảnh đó thuộc Cluster số mấy. Kết quả được đánh giá chéo qua biểu đồ Loss/Accuracy và Confusion Matrix.

> **Tiến độ hiện tại:** Phase 1-3 đã hoàn thành. Web app đã chạy được với các tính năng của 3 phase này. Phase 4–5 đang tiếp tục phát triển.

## 4. Cấu trúc thư mục

```text
KhaiPhaDuLieu-main/
 ┣ models/                       # File model đã huấn luyện (.keras, .pth) và file K-Means (.pkl)
 ┣ notebooks/                    # Source code huấn luyện mô hình (Colab / Jupyter Notebooks)
 ┣ src/                          # Mã nguồn ứng dụng Web (Gradio Dashboard)
 ┃ ┣ app_main.py                 # File entry point chạy ứng dụng chính (Dashboard với Sidebar)
 ┃ ┣ tab_phase1.py               # Giao diện chạy Baseline Model
 ┃ ┣ tab_phase2.py               # Giao diện chạy Augmented Model
 ┃ ┣ tab_phase3.py               # Giao diện chạy CutMix Model (PyTorch ResNet18)
 ┃ ┣ tab_mosaic.py               # Giao diện chạy Mosaic Model (PyTorch ResNet18)
 ┃ ┣ tab_augmentation.py         # Demo các phép tăng cường trên 1 ảnh trực quan
 ┃ ┣ tab_final_app.py            # Tab ứng dụng tích hợp Phân loại & Phân cụm K-Means
 ┃ ┣ tab_batch_inference.py      # Phân loại hàng loạt nhiều ảnh, gom nhóm xuất file zip kết quả
 ┃ ┣ tab_batch_augmentation.py  # "Trạm Tăng Cường Dữ Liệu" - Nhân bản dataset tự động
 ┃ ┣ aug_basic.py, aug_cutmix.py, aug_mosaic.py, aug_helpers.py # Các hàm biến đổi ảnh nâng cao
 ┃ ┣ analyze_clusters.py         # Phân tích gom cụm K-Means
 ┃ ┗ cluster_augmentation.py     # Hỗ trợ tăng cường theo cluster
 ┣ AnhTest/                      # Các ảnh thử nghiệm nhanh
 ┣ TepTest/                      # Thư mục chứa dataset kiểm thử theo lớp
 ┣ README.md                     # Tài liệu hướng dẫn sử dụng (File này)
 ┗ requirements.txt              # Danh sách thư viện bắt buộc
```

---

## 5. Hướng dẫn Cài đặt & Chạy ứng dụng

### Bước 1: Tạo môi trường ảo Python
Sử dụng PowerShell trên Windows:
```powershell
python -m venv KPDL
.\KPDL\Scripts\Activate.ps1
```

### Bước 2: Nâng cấp pip và cài đặt thư viện
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

*(Lưu ý: Môi trường ảo cũng yêu cầu thư viện `pandas` và `opencv-python` đã được thêm đầy đủ vào `requirements.txt`).*

### Bước 3: Khởi chạy ứng dụng
Từ thư mục gốc của dự án, chạy lệnh:
```powershell
python src/app_main.py
```
Sau khi khởi động xong, truy cập vào đường dẫn cục bộ hiển thị trên console (mặc định là **`http://127.0.0.1:7860`**).

---

## 6. Hướng dẫn sử dụng các tính năng đặc sắc

### 6.1 Giao diện Dashboard hiện đại
Ứng dụng sử dụng Light Theme sáng màu, chuyên nghiệp với Sidebar bên trái cố định để chuyển đổi các tính năng:
- **Phase 1 đến Phase 4**: Demo suy luận của từng mô hình đơn lẻ (Baseline, Augmented, CutMix, Mosaic).
- **Data Augmentation Tool**: Thử nghiệm trực quan các phép biến đổi trên 1 ảnh.
- **Final Application**: Phân loại ảnh kết hợp rút trích đặc trưng phân cụm K-Means để giải thích lý do phân loại.
- **Batch Inference**: Kéo thả thư mục/danh sách nhiều ảnh để dự đoán hàng loạt, kết xuất bảng dữ liệu và chia tệp ZIP kết quả theo nhãn dự đoán.

### 6.2 Trạm Tăng Cường Dữ Liệu (Augmentation Station)
Đây là công cụ mạnh mẽ dành cho việc làm giàu tập dữ liệu huấn luyện:
1. **Đầu vào (Upload):** Chấp nhận tải lên trực tiếp danh sách nhiều ảnh rời hoặc một tệp nén **`dataset.zip`** có cấu trúc thư mục phân loại theo lớp (ví dụ: `Burger/`, `Pizza/`, ...).
2. **Cấu hình tăng cường:**
   - **Kỹ thuật cơ bản (Safe):** Flip (lật ngang dọc), Rotate (xoay nhẹ an toàn từ -15° đến 15°), Zoom/Crop nhẹ (90% - 110%), Color Jitter (chỉnh màu sắc/độ sáng nhẹ), Gaussian Blur.
   - **Kỹ thuật nâng cao (Advanced):** CutMix, Mosaic (có cảnh báo có thể làm sai nhãn do trộn lớp, khuyên dùng cho demo).
   - **Số lần nhân bản (copies):** Điều chỉnh trực quan bằng nút cộng `+`/ trừ `-`.
3. **Đầu ra (Output):**
   - Tạo ra tệp **`Augmented_Dataset.zip`** tải về ngay trên UI.
   - File ZIP kết quả giữ nguyên cấu trúc thư mục lớp ban đầu.
   - Tự động sinh tệp tin báo cáo **`augmentation_report.csv`** nằm bên trong tệp ZIP chứa đầy đủ lịch sử tăng cường: `class`, `original_file`, `augmented_file`, `technique`, `mode`.
4. **Giới hạn hệ thống:** Giới hạn tối đa xử lý 100 ảnh gốc và tạo tối đa 1000 ảnh đầu ra để tránh quá tải CPU/RAM trên máy chủ. Có hiển thị thông báo/cảnh báo rõ ràng trên UI nếu đạt ngưỡng.
