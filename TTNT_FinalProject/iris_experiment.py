#type: ignore
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.decomposition import PCA
from ucimlrepo import fetch_ucirepo

# ==================== KHỞI TẠO ==========================================
print("\n================== TRIỂN KHAI K-MEANS TRÊN BỘ DỮ LIỆU IRIS (KHÔNG CÓ NHÃN) ===========================")

# Tạo thư mục output
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
iris_output_dir = os.path.join(project_root, 'irisoutput')
os.makedirs(iris_output_dir, exist_ok=True)

# ==================== 5.1.1. LOAD DỮ LIỆU VÀ TIỀN XỬ LÝ ====================
print("\n=== 5.1.1. LOAD DỮ LIỆU VÀ TIỀN XỬ LÝ ===")

# Load dữ liệu không có nhãn
try:
    dataset_dir = os.path.join(current_dir, 'dataset')
    features_file = os.path.join(dataset_dir, 'iris_features_only.xlsx')
    X_unlabeled = pd.read_excel(features_file)
    print(f"✓ Đã đọc dữ liệu không có nhãn: {X_unlabeled.shape} mẫu")
except Exception as e:
    print(f"✗ Lỗi khi đọc file: {e}")
    exit()

# Tải bộ dữ liệu gốc để lấy nhãn thực tế (chỉ dùng để đánh giá sau này)
iris = fetch_ucirepo(id=53)
X_original = iris.data.features
y_original = iris.data.targets['class'].values

# Tiền xử lý - Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_unlabeled)
print("✓ Đã chuẩn hóa dữ liệu")

# ==================== 5.1.2. ÁP DỤNG K-MEANS ====================
print("\n=== 5.1.2. ÁP DỤNG K-MEANS ===")

# Elbow Method
wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, marker='o', linestyle='-')
plt.title('Elbow Method cho dataset Iris')
plt.xlabel('Số cụm (k)')
plt.ylabel('WCSS')
plt.grid(True, alpha=0.3)
elbow_file = os.path.join(iris_output_dir, 'iris_elbow_method.png')
plt.savefig(elbow_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Đã lưu biểu đồ Elbow Method: {elbow_file}")

# Silhouette Analysis
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"k={k}: Silhouette Score = {score:.3f}")

plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='-')
plt.title('Silhouette Analysis cho dataset Iris')
plt.xlabel('Số cụm (k)')
plt.ylabel('Silhouette Score')
plt.grid(True, alpha=0.3)
silhouette_file = os.path.join(iris_output_dir, 'iris_silhouette_analysis.png')
plt.savefig(silhouette_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Đã lưu biểu đồ Silhouette Analysis: {silhouette_file}")

# Tìm k tối ưu từ Silhouette Analysis
k_optimal = np.argmax(silhouette_scores) + 2  # vì silhouette_scores bắt đầu từ k=2
print(f"\nÁp dụng K-Means với k={k_optimal} (số cụm tối ưu rút ra từ Silhouette Analysis)")
kmeans_final = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)

# ==================== 5.1.3. THU THẬP KẾT QUẢ ====================
print("\n=== 5.1.3. THU THẬP KẾT QUẢ ===")

# Thêm nhãn phân cụm vào dữ liệu gốc
df_result = X_unlabeled.copy()
df_result['cluster'] = cluster_labels

# Thống kê số lượng mẫu trong từng cụm
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
print("Số lượng mẫu trong từng cụm:")
for cluster, count in cluster_counts.items():
    print(f"  Cụm {cluster}: {count} mẫu")

# Tâm cụm
centroids = kmeans_final.cluster_centers_
centroids_original = scaler.inverse_transform(centroids)
print("\nTâm cụm (giá trị gốc):")
for i, centroid in enumerate(centroids_original):
    print(f"  Cụm {i}: {centroid}")

# Lưu kết quả phân cụm
result_file = os.path.join(iris_output_dir, 'iris_clustered.xlsx')
df_result.to_excel(result_file, index=False)
print(f"✓ Đã lưu kết quả phân cụm: {result_file}")

# ==================== 5.2.1. SO SÁNH VỚI NHÃN THỰC TẾ ====================
print("\n=== 5.2.1. SO SÁNH VỚI NHÃN THỰC TẾ ===")

# Chuyển nhãn thực tế thành số
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_true = le.fit_transform(y_original)

# Tính các chỉ số đánh giá
ari = adjusted_rand_score(y_true, cluster_labels)
nmi = normalized_mutual_info_score(y_true, cluster_labels)

print(f"Adjusted Rand Index (ARI): {ari:.3f}")
print(f"Normalized Mutual Information (NMI): {nmi:.3f}")

# Tạo bảng so sánh nhãn thực tế và kết quả phân cụm
comparison_table = pd.crosstab(
    pd.Series(y_true, name='Nhãn thực tế'),
    pd.Series(cluster_labels, name='Cluster')
)
print("\nBảng so sánh nhãn thực tế và kết quả phân cụm:")
print(comparison_table)

# ==================== 5.2.2. ĐÁNH GIÁ NỘI BỘ ====================
print("\n=== 5.2.2. ĐÁNH GIÁ NỘI BỘ ===")

# Silhouette Score tổng thể
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
print(f"Silhouette Score: {silhouette_avg:.3f}")

# ==================== 5.2.3. TRỰC QUAN HÓA ====================
print("\n=== 5.2.3. TRỰC QUAN HÓA ===")

# Trực quan hóa dựa trên PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Vẽ kết quả phân cụm trên không gian PCA
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.8)
plt.title('K-Means Clustering trên Iris (PCA)', fontsize=14)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)

# Thêm tâm cụm
centroids_pca = pca.transform(centroids)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
           s=200, marker='X', c='red', edgecolors='black', linewidths=2, label='Centroids')

plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)
plt.legend()
pca_file = os.path.join(iris_output_dir, 'iris_kmeans_pca.png')
plt.savefig(pca_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Đã lưu biểu đồ PCA: {pca_file}")

# Trực quan hóa dựa trên 2 đặc trưng quan trọng
plt.figure(figsize=(10, 8))
petal_length_col = [col for col in X_unlabeled.columns if 'petal length' in col][0]
petal_width_col = [col for col in X_unlabeled.columns if 'petal width' in col][0]
plt.scatter(
    X_unlabeled[petal_length_col], X_unlabeled[petal_width_col], 
    c=cluster_labels, cmap='viridis', s=50, alpha=0.8
)
plt.title('K-Means Clustering trên Iris (Petal Length vs Width)', fontsize=14)
plt.xlabel(petal_length_col, fontsize=12)
plt.ylabel(petal_width_col, fontsize=12)

# Thêm tâm cụm
for i, centroid in enumerate(centroids_original):
    plt.scatter(
        centroid[2], centroid[3], 
        s=200, marker='X', c='red', edgecolor='black', linewidth=2
    )
    plt.annotate(f'Cluster {i}', (centroid[2], centroid[3]), 
                fontsize=12, xytext=(10, 5), textcoords='offset points')

plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)
features_file = os.path.join(iris_output_dir, 'iris_kmeans_features.png')
plt.savefig(features_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Đã lưu biểu đồ phân tán đặc trưng: {features_file}")

# So sánh nhãn thực tế và nhãn phân cụm
plt.figure(figsize=(12, 5))

# Vẽ biểu đồ thứ nhất: phân cụm theo K-Means
plt.subplot(1, 2, 1)
scatter1 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.8)
plt.title('Phân cụm K-Means', fontsize=14)
plt.xlabel('PC1', fontsize=12)
plt.ylabel('PC2', fontsize=12)
plt.colorbar(scatter1, label='Cluster')

# Vẽ biểu đồ thứ hai: nhãn thực tế
plt.subplot(1, 2, 2)
scatter2 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='viridis', s=50, alpha=0.8)
plt.title('Nhãn thực tế', fontsize=14)
plt.xlabel('PC1', fontsize=12)
plt.ylabel('PC2', fontsize=12)
plt.colorbar(scatter2, label='True Label')

plt.tight_layout()
compare_file = os.path.join(iris_output_dir, 'iris_kmeans_vs_true.png')
plt.savefig(compare_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Đã lưu biểu đồ so sánh: {compare_file}")

# ==================== KẾT LUẬN ====================
print("\n=== KẾT LUẬN THỰC NGHIỆM ===")
print(f"1. K-Means đã phân cụm thành công dataset Iris thành {k_optimal} cụm.")
print(f"2. Silhouette Score = {silhouette_avg:.3f} cho thấy chất lượng phân cụm tốt.")
print(f"3. ARI = {ari:.3f}, NMI = {nmi:.3f} cho thấy độ phù hợp cao với nhãn thực tế.")
print("4. Các biểu đồ trực quan hóa đã được lưu trong thư mục irisoutput.")
