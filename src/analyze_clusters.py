import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image
from sklearn.preprocessing import normalize


MODEL_P1_PATH = "models/baseline_model_p1.keras"
KMEANS_PATH = "models/kmeans_model.pkl"

# Sửa dòng này theo đúng thư mục train dataset trên máy bạn
DATASET_DIR = "data/train"

IMAGE_SIZE = (150, 150)

LABELS = [
    "Baked Potato", "Burger", "Crispy Chicken", "Donut", "Fries",
    "Hot Dog", "Pizza", "Sandwich", "Taco", "Taquito"
]


print("Đang load model Phase 1...")
model_p1 = tf.keras.models.load_model(MODEL_P1_PATH)

print("Đang tạo feature extractor từ Phase 1...")
feature_extractor = tf.keras.Sequential(
    model_p1.layers[:-1],
    name="feature_extractor_p1"
)

dummy_input = np.zeros((1, 150, 150, 3), dtype=np.float32)
dummy_feature = feature_extractor.predict(dummy_input, verbose=0)
print("Feature extractor output shape:", dummy_feature.shape)

print("Đang load K-Means...")
kmeans = joblib.load(KMEANS_PATH)
print("Số cụm:", kmeans.n_clusters)
print("Số chiều K-Means cần:", kmeans.n_features_in_)


image_paths = []
image_labels = []

for label in LABELS:
    class_dir = os.path.join(DATASET_DIR, label)

    if not os.path.exists(class_dir):
        print(f"Cảnh báo: Không tìm thấy thư mục {class_dir}")
        continue

    for file_name in os.listdir(class_dir):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            image_paths.append(os.path.join(class_dir, file_name))
            image_labels.append(label)

print(f"Tổng số ảnh đọc được: {len(image_paths)}")

if len(image_paths) == 0:
    raise ValueError("Không đọc được ảnh nào. Hãy kiểm tra lại DATASET_DIR.")


records = []

for path, label in zip(image_paths, image_labels):
    try:
        img = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
        img_array = np.expand_dims(np.array(img), axis=0)

        feature = feature_extractor.predict(img_array, verbose=0)

        if feature.shape[1] != kmeans.n_features_in_:
            raise ValueError(
                f"Feature có {feature.shape[1]} chiều, "
                f"nhưng K-Means cần {kmeans.n_features_in_} chiều"
            )

        feature_norm = normalize(feature)
        cluster_id = int(kmeans.predict(feature_norm)[0])

        records.append({
            "image_path": path,
            "label": label,
            "cluster": cluster_id
        })

    except Exception as e:
        print(f"Lỗi xử lý ảnh {path}: {e}")


df = pd.DataFrame(records)

print("\n===== BẢNG CLUSTER - LABEL =====")
cluster_table = pd.crosstab(df["cluster"], df["label"])
print(cluster_table)

summary = []

for cluster_id in sorted(df["cluster"].unique()):
    subset = df[df["cluster"] == cluster_id]
    counts = subset["label"].value_counts()

    main_label = counts.index[0]
    main_count = counts.iloc[0]
    total = len(subset)
    percent = round(main_count / total * 100, 2)

    summary.append({
        "cluster": cluster_id,
        "main_label": main_label,
        "main_count": main_count,
        "total_images": total,
        "percent": percent
    })

summary_df = pd.DataFrame(summary)

print("\n===== NHÃN CHIẾM ĐA SỐ MỖI CLUSTER =====")
print(summary_df)

os.makedirs("outputs", exist_ok=True)

df.to_csv("outputs/cluster_assignments.csv", index=False, encoding="utf-8-sig")
summary_df.to_csv("outputs/cluster_summary.csv", index=False, encoding="utf-8-sig")

print("\nĐã xuất file:")
print("outputs/cluster_assignments.csv")
print("outputs/cluster_summary.csv")