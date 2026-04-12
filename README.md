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

> **Tiến độ hiện tại:** Phase 1 và Phase 2 đã hoàn thành. Web app đã chạy được với các tính năng của 2 phase này. Phase 3–5 đang tiếp tục phát triển.

---

## 4. Cấu trúc thư mục

```text
Do_An_Khai_Pha_Du_Lieu/
 ┣ models/                       # Nơi chứa file model (.keras) và file K-Means (.pkl)
 ┣ notebooks/                    # Source code quá trình train trên Google Colab
 ┣ src/                          # Mã nguồn ứng dụng Web
 ┃ ┣ augmentor.py                # Thuật toán tăng cường dữ liệu
 ┃ ┣ tab_phase1.py               # Module chạy Baseline
 ┃ ┣ tab_phase2.py               # Module chạy Model Nâng cao
 ┃ ┣ tab_augmentation.py         # Module demo sinh ảnh trực tiếp
 ┃ ┗ app_main.py                 # File gốc để chạy web (Gradio)
 ┣ README.md                     # Tài liệu hướng dẫn
 ┗ requirements.txt              # Danh sách thư viện
```

---

## 5. Cài đặt thư viện
pip install tensorflow scikit-learn gradio numpy pillow matplotlib

