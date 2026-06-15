import os
import sys
import pandas as pd
from PIL import Image

# Cho phép import file trong src/
sys.path.append("src")

from aug_basic import pil_basic_augment


# File kết quả sau khi phân tích K-Means
CLUSTER_ASSIGNMENTS_PATH = "outputs/cluster_assignments.csv"

# Các cluster khó được chọn từ cluster_summary.csv
TARGET_CLUSTERS = [4, 1, 7]

# Mỗi ảnh gốc sinh bao nhiêu ảnh augmentation
AUG_PER_IMAGE = 3

# Giới hạn số ảnh xử lý mỗi cluster để tránh tạo quá nhiều ảnh
MAX_IMAGES_PER_CLUSTER = 100

# Thư mục lưu ảnh sinh ra
OUTPUT_DIR = "Data_Cluster_Augmented"


def main():
    if not os.path.exists(CLUSTER_ASSIGNMENTS_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy {CLUSTER_ASSIGNMENTS_PATH}. "
            "Hãy chạy scripts/analyze_clusters.py trước."
        )

    df = pd.read_csv(CLUSTER_ASSIGNMENTS_PATH)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_created = 0

    for cluster_id in TARGET_CLUSTERS:
        cluster_df = df[df["cluster"] == cluster_id].copy()

        if len(cluster_df) == 0:
            print(f"Cluster {cluster_id} không có ảnh, bỏ qua.")
            continue

        cluster_df = cluster_df.sample(
            n=min(MAX_IMAGES_PER_CLUSTER, len(cluster_df)),
            random_state=42
        )

        print(f"\nĐang xử lý Cluster {cluster_id}: {len(cluster_df)} ảnh gốc")

        for _, row in cluster_df.iterrows():
            image_path = row["image_path"]
            label = row["label"]

            if not os.path.exists(image_path):
                print(f"Không tìm thấy ảnh: {image_path}")
                continue

            label_output_dir = os.path.join(
                OUTPUT_DIR,
                f"cluster_{cluster_id}",
                label
            )
            os.makedirs(label_output_dir, exist_ok=True)

            try:
                img = Image.open(image_path).convert("RGB")
                base_name = os.path.splitext(os.path.basename(image_path))[0]

                for i in range(AUG_PER_IMAGE):
                    aug_img = pil_basic_augment(
                        img.copy(),
                        rotation_range=0.15,
                        zoom_range=0.1,
                        flip_h=True,
                        flip_v=False
                    )

                    out_name = f"{base_name}_cluster{cluster_id}_aug{i}.jpg"
                    out_path = os.path.join(label_output_dir, out_name)

                    aug_img.save(out_path, quality=95)
                    total_created += 1

            except Exception as e:
                print(f"Lỗi xử lý ảnh {image_path}: {e}")

    print("\nHoàn thành cluster-guided basic augmentation.")
    print(f"Tổng số ảnh augmented đã tạo: {total_created}")
    print(f"Ảnh lưu tại thư mục: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()