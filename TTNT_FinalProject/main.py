# type: ignore
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#----------------------------------------------------------------------------
# Bước 1: Chuẩn bị dữ liệu
irispk = load_iris()  # Tải bộ dữ liệu Iris từ sklearn
x = pd.DataFrame(irispk.data, columns=irispk.feature_names)  # Đặc trưng (features)
y = pd.Series(irispk.target)  # Nhãn thực tế (labels/classes)

#----------------------------------------------------------------------------
# Bước 2: KMeans
k = 3  # Số cụm cần gom, bằng số lớp thực tế của Iris
kmeans = KMeans(n_clusters=k, random_state=42)  # Khởi tạo mô hình KMeans
kmeans.fit(x)  # Huấn luyện mô hình trên dữ liệu
labels = kmeans.labels_  # Nhãn cụm dự đoán cho từng mẫu

#----------------------------------------------------------------------------
# Bước 3: Đánh giá
# Sử dụng các tiêu chí đánh giá ngoại để so sánh nhãn gom cụm với nhãn thực tế
ari = adjusted_rand_score(y, labels)  # Chỉ số Rand hiệu chỉnh
nmi = normalized_mutual_info_score(y, labels)  # Thông tin tương hỗ chuẩn hóa
homogeneity = homogeneity_score(y, labels)  # Độ thuần nhất
completeness = completeness_score(y, labels)  # Độ đầy đủ
v_measure = v_measure_score(y, labels)  # Trung bình điều hòa giữa độ thuần nhất và đầy đủ

print(f"Adjusted Rand Index (ARI): {ari:.3f}")
print(f"Normalized Mutual Information (NMI): {nmi:.3f}")
print(f"Homogeneity: {homogeneity:.3f}")
print(f"Completeness: {completeness:.3f}")
print(f"V-Measure: {v_measure:.3f}")

#----------------------------------------------------------------------------
# Bước 4: Trực quan hóa (tùy chọn)
# Giảm chiều dữ liệu xuống 2D bằng PCA để dễ vẽ biểu đồ
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.scatter(x_pca[:,0], x_pca[:,1], c=y, cmap='viridis', s=30)
plt.title('Ground Truth')  # Biểu đồ nhãn thực tế

plt.subplot(1,2,2)
plt.scatter(x_pca[:,0], x_pca[:,1], c=labels, cmap='viridis', s=30)
plt.title('KMeans Clusters')  # Biểu đồ nhãn gom cụm KMeans

plt.show()