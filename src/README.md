# ĐỒ ÁN: PHÂN LOẠI VÀ TĂNG CƯỜNG DỮ LIỆU ẢNH THỰC PHẨM
**Học phần:** Khai phá dữ liệu (Data Mining)  
**Đề tài 5:** Xây dựng ứng dụng tăng cường dữ liệu ảnh  
**Dataset:** Fast Food Data (10 nhãn lớp)

---

## 1. Giới thiệu tổng quan
Dự án tập trung vào việc tối ưu hóa khả năng nhận diện hình ảnh thông qua các kỹ thuật tăng cường dữ liệu (Data Augmentation) và phân tích cấu trúc dữ liệu bằng thuật toán phân cụm (Clustering). Hệ thống được xây dựng trên nền tảng mạng nơ-ron tích chập (CNN) và giao diện tương tác Gradio.

---

## 2. Vai trò của các thành phần (P2, P4)

| Thành phần | Vai trò | Chức năng cụ thể |
| :--- | :--- | :--- |
| **CNN Model** | **Chủ đạo** | Thực hiện phân loại ảnh thực phẩm dựa trên các đặc trưng học được từ tập dữ liệu. |
| **K-Means** | **Hỗ trợ (Support)** | Sử dụng Feature Vector từ Phase 1 để gom nhóm ảnh. Giúp phân tích độ tương đồng và các trường hợp mô hình dự đoán nhầm. |
| **Augmentation** | **Tăng cường** | Áp dụng cơ chế **On-the-fly** (biến đổi ngay khi train) để tăng tính tổng quát hóa, chống hiện tượng Overfitting. |

---

## 3. Cơ sở lý thuyết và Công thức (P2, P3)

### 3.1. Phân cụm K-Means (Support Layer)
Thuật toán tối thiểu hóa tổng bình phương sai số (SSE) dựa trên khoảng cách Euclidean:
$$SSE = \sum_{j=1}^{k} \sum_{x \in C_j} \|x - c_j\|^2$$

### 3.2. Kỹ thuật Tăng cường dữ liệu
* **Cơ bản (Phase 2):** Bao gồm các phép biến đổi hình học như Flip, Rotate, và Crop. Ma trận xoay ảnh góc $\theta$:
$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}$$
* **Nâng cao (Phase 3):** Áp dụng **MixUp** - kết hợp hai hình ảnh ngẫu nhiên để tạo ra dữ liệu mới:
$$\tilde{x} = \lambda x_i + (1 - \lambda) x_j$$

---

## 4. Quy trình thực hiện (Pipeline)

* **Phase 1 (Baseline):** Chuẩn hóa (Normalize), Resize (150x150) và huấn luyện CNN thuần (không Augmentation). Trích xuất Feature Vector từ lớp ẩn.
* **Phase 2 (Clustering & Basic Aug):** Phân cụm đặc trưng bằng K-Means. Xây dựng module Augmentation "On-the-fly" với các phép biến đổi cơ bản.
* **Phase 3 (Advanced Aug):** Triển khai kỹ thuật nâng cao (MixUp/CutMix). Thực hiện đánh giá so sánh hiệu năng qua biểu đồ Loss và Accuracy giữa các Phase.
* **Phase 4 (App Integration):** Tích hợp Model vào ứng dụng Gradio. Hiển thị thông tin dự đoán kèm theo định danh Cluster để phân tích sâu.
* **Phase 5 (Analysis):** Tổng hợp báo cáo, đánh giá dựa trên Confusion Matrix và so sánh trực quan giữa các kịch bản huấn luyện.

---

## 5. Cấu trúc thư mục
```text
Do_An_Khai_Pha_Du_Lieu/
 ┣ models/                       # Chứa file mô hình (.keras) và K-Means (.pkl)
 ┣ src/                          # Mã nguồn ứng dụng Web
 ┃ ┣ augmentor.py                # Module Augmentation dùng chung
 ┃ ┣ tab_phase1.py               # Giao diện kiểm thử Baseline
 ┃ ┣ tab_phase2.py               # Giao diện kiểm thử Advanced Model
 ┃ ┣ tab_augmentation.py         # Demo trực quan hóa sinh ảnh
 ┃ ┗ app_main.py                 # File thực thi chính khởi chạy Gradio
 ┗ README.md                     # Tài liệu hướng dẫn