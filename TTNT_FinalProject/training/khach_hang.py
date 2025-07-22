# type: ignore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os
from math import pi

# Đặt matplotlib backend để tránh lỗi hiển thị
plt.switch_backend('Agg')

# ========== KHỞI TẠO VÀ KIỂM TRA THỦ MỤC ==========
print("=== BẮT ĐẦU PHÂN TÍCH PHÂN CỤM KHÁCH HÀNG ===")
print("Thư mục hiện tại:", os.getcwd())

# Tạo thư mục output với đường dẫn tuyệt đối
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
output_dir = os.path.join(project_root, 'output')

try:
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ Thư mục output đã được tạo: {output_dir}")
except Exception as e:
    print(f"✗ Lỗi khi tạo thư mục output: {e}")
    exit()

# Kiểm tra quyền ghi
try:
    test_file = os.path.join(output_dir, 'test.txt')
    with open(test_file, 'w') as f:
        f.write('test')
    os.remove(test_file)
    print("✓ Thư mục output có quyền ghi")
except Exception as e:
    print(f"✗ Không có quyền ghi vào thư mục output: {e}")
    exit()

# ========== ĐỌC DỮ LIỆU ==========
print("\n=== BƯỚC 1: ĐỌC DỮ LIỆU ===")

# Tìm file Excel
dataset_dir = os.path.join(project_root, 'dataset')
possible_paths = [
    os.path.join(dataset_dir, 'khach_hang_dataset.xlsx'),
    os.path.join(current_dir, '..', 'dataset', 'khach_hang_dataset.xlsx'),
    os.path.join(current_dir, 'dataset', 'khach_hang_dataset.xlsx')
]

excel_path = None
for path in possible_paths:
    if os.path.exists(path):
        excel_path = path
        print(f"✓ Tìm thấy file tại: {path}")
        break

if excel_path is None:
    print("⚠ Không tìm thấy file Excel. Tạo dữ liệu mẫu...")
    try:
        # Tạo dữ liệu mẫu
        np.random.seed(42)
        data = {
            'Tuổi': np.random.randint(18, 65, 100),
            'Thu nhập (USD/năm)': np.random.randint(20000, 120000, 100),
            'Số lần mua hàng/năm': np.random.randint(1, 40, 100),
            'Tổng chi tiêu (USD/năm)': np.random.randint(200, 16000, 100)
        }
        df = pd.DataFrame(data)
        
        # Tạo thư mục dataset nếu chưa tồn tại
        os.makedirs(dataset_dir, exist_ok=True)
        sample_file = os.path.join(dataset_dir, 'khach_hang_dataset.xlsx')
        df.to_excel(sample_file, index=False)
        print(f"✓ Đã tạo file dữ liệu mẫu tại: {sample_file}")
        excel_path = sample_file
    except Exception as e:
        print(f"✗ Lỗi khi tạo dữ liệu mẫu: {e}")
        exit()

# Đọc dữ liệu
try:
    df = pd.read_excel(excel_path)
    print(f"✓ Đã đọc thành công {len(df)} dòng dữ liệu")
except Exception as e:
    print(f"✗ Lỗi khi đọc file Excel: {e}")
    exit()

print("\nThông tin dữ liệu:")
print(df.head())
print("\nTên cột thực tế:", df.columns.tolist())
print("\nKiểm tra missing values:")
print(df.isnull().sum())

# ========== CHUẨN BỊ DỮ LIỆU =========================
print("\n=== BƯỚC 2: CHUẨN BỊ DỮ LIỆU ===")

# Chuẩn hóa tên cột------------------------------------
original_columns = df.columns.tolist()
if 'Thu nhập (USD/năm)' in df.columns:
    df.rename(columns={
        'Thu nhập (USD/năm)': 'Thu nhập',
        'Tổng chi tiêu (USD/năm)': 'Tổng chi tiêu'
    }, inplace=True)
    print("✓ Đã chuẩn hóa tên cột")

# Định nghĩa các đặc trưng-----------------------------
features = ['Tuổi', 'Thu nhập', 'Số lần mua hàng/năm', 'Tổng chi tiêu']
print(f"Sử dụng các đặc trưng: {features}")

# Kiểm tra các cột có tồn tại--------------------------
missing_features = [f for f in features if f not in df.columns]
if missing_features:
    print(f"✗ Các cột bị thiếu: {missing_features}")
    print(f"Các cột có sẵn: {df.columns.tolist()}")
    exit()

print("✓ Tất cả các đặc trưng đều có trong dữ liệu")

# Chuẩn bị ma trận đặc trưng---------------------------
try:
    X = df[features].values
    print(f"✓ Ma trận đặc trưng có kích thước: {X.shape}")
except Exception as e:
    print(f"✗ Lỗi khi tạo ma trận đặc trưng: {e}")
    exit()

# Chuẩn hóa dữ liệu------------------------------------
try:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✓ Đã chuẩn hóa dữ liệu")
except Exception as e:
    print(f"✗ Lỗi khi chuẩn hóa dữ liệu: {e}")
    exit()

# ========== XÁC ĐỊNH SỐ CỤM TỐI ƯU ===================
print("\n=== BƯỚC 3: XÁC ĐỊNH SỐ CỤM TỐI ƯU ===")

# Elbow method-----------------------------------------
try:
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='-', linewidth=2, markersize=6)
    plt.title('Elbow Method - Xác định số cụm tối ưu', fontsize=14)
    plt.xlabel('Số cụm (k)', fontsize=12)
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    elbow_file = os.path.join(output_dir, 'elbow_method.png')
    plt.savefig(elbow_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Đã lưu biểu đồ Elbow Method: {elbow_file}")
except Exception as e:
    print(f"✗ Lỗi khi tạo biểu đồ Elbow Method: {e}")

# Silhouette analysis----------------------------------
try:
    silhouette_scores = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"Với {k} cụm, silhouette score là {silhouette_avg:.3f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='-', linewidth=2, markersize=6)
    plt.title('Silhouette Analysis - Đánh giá chất lượng phân cụm', fontsize=14)
    plt.xlabel('Số cụm (k)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    silhouette_file = os.path.join(output_dir, 'silhouette_analysis.png')
    plt.savefig(silhouette_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Đã lưu biểu đồ Silhouette Analysis: {silhouette_file}")
    
    # Tìm k tối ưu---------------------------------------
    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
    print(f"✓ Số cụm tối ưu theo Silhouette Score: {optimal_k}")
except Exception as e:
    print(f"✗ Lỗi khi tạo biểu đồ Silhouette Analysis: {e}")
    optimal_k = 3

# ========== ÁP DỤNG K-MEANS ============================
print(f"\n=== BƯỚC 4: ÁP DỤNG K-MEANS VỚI K={optimal_k} ===")

try:
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    print(f"✓ Đã phân cụm thành công với {optimal_k} cụm")
    
    # Thống kê cụm
    cluster_counts = df['Cluster'].value_counts().sort_index()
    print("Số lượng khách hàng trong mỗi cụm:")
    for cluster, count in cluster_counts.items():
        print(f"  Cụm {cluster}: {count} khách hàng")
        
except Exception as e:
    print(f"✗ Lỗi khi áp dụng K-Means: {e}")
    exit()

# ========== PHÂN TÍCH CÁC CỤM ==========
print("\n=== BƯỚC 5: PHÂN TÍCH CÁC CỤM ===")

print("Đặc điểm trung bình của các cụm:")
for cluster in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster][features]
    print(f"\nCụm {cluster} ({len(cluster_data)} khách hàng):")
    print(cluster_data.describe().round(2))

# ========== TRỰC QUAN HÓA ==========
print("\n=== BƯỚC 6: TRỰC QUAN HÓA KẾT QUẢ ===")

# Hàm lưu biểu đồ an toàn
def save_plot_safe(filename, description):
    try:
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Đã lưu {description}: {filepath}")
        return True
    except Exception as e:
        print(f"✗ Lỗi khi lưu {description}: {e}")
        plt.close()
        return False

# 1. Scatter plot các cặp đặc trưng
try:
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = axs.flatten()
    
    pairs = [
        ('Tuổi', 'Thu nhập'),
        ('Tuổi', 'Số lần mua hàng/năm'),
        ('Tuổi', 'Tổng chi tiêu'),
        ('Thu nhập', 'Số lần mua hàng/năm'),
        ('Thu nhập', 'Tổng chi tiêu'),
        ('Số lần mua hàng/năm', 'Tổng chi tiêu')
    ]
    
    centroids = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids)
    
    for i, (x_feature, y_feature) in enumerate(pairs):
        sns.scatterplot(x=x_feature, y=y_feature, hue='Cluster', 
                       data=df, palette='viridis', ax=axs[i], alpha=0.7)
        axs[i].set_title(f'{x_feature} vs {y_feature}', fontsize=12)
        
        # Đánh dấu tâm cụm
        axs[i].scatter(
            centroids_original[:, features.index(x_feature)],
            centroids_original[:, features.index(y_feature)],
            s=300, marker='X', c='red', edgecolors='black', linewidths=2
        )
        
        if i == 0:
            axs[i].legend(title='Cụm', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    save_plot_safe('cluster_pairs.png', 'biểu đồ scatter các cặp đặc trưng')
    
except Exception as e:
    print(f"✗ Lỗi khi tạo biểu đồ scatter pairs: {e}")

# 2. PCA visualization
try:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], 
                         cmap='viridis', s=50, alpha=0.8)
    plt.title('PCA - Phân cụm khách hàng trong không gian 2D', fontsize=14)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    
    # Thêm tâm cụm
    centroids_pca = pca.transform(centroids)
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
               s=300, marker='X', c='red', edgecolors='black', linewidths=2)
    
    plt.colorbar(scatter, label='Cụm')
    plt.grid(True, alpha=0.3)
    save_plot_safe('pca_clusters.png', 'biểu đồ PCA 2D')
    
except Exception as e:
    print(f"✗ Lỗi khi tạo biểu đồ PCA: {e}")

# 3. 3D scatter plot
try:
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter_3d = ax.scatter(df['Tuổi'], df['Thu nhập'], df['Tổng chi tiêu'],
                           c=df['Cluster'], cmap='viridis', alpha=0.6, s=50)
    ax.set_xlabel('Tuổi', fontsize=12)
    ax.set_ylabel('Thu nhập', fontsize=12)
    ax.set_zlabel('Tổng chi tiêu', fontsize=12)
    ax.set_title('Phân cụm khách hàng 3D', fontsize=14)
    
    # Thêm tâm cụm
    for i, centroid in enumerate(centroids_original):
        ax.scatter(centroid[0], centroid[1], centroid[3],
                  s=300, marker='X', c='red', edgecolors='black', linewidths=2)
    
    plt.colorbar(scatter_3d, label='Cụm', shrink=0.5)
    save_plot_safe('3d_clusters.png', 'biểu đồ 3D')
    
except Exception as e:
    print(f"✗ Lỗi khi tạo biểu đồ 3D: {e}")

# 4. Biểu đồ cột so sánh
try:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, feature in enumerate(features):
        cluster_means = df.groupby('Cluster')[feature].mean()
        bars = axes[i].bar(range(len(cluster_means)), cluster_means.values,
                          color=colors[:len(cluster_means)], alpha=0.8)
        axes[i].set_title(f'Trung bình {feature} theo cụm', fontsize=12)
        axes[i].set_xlabel('Cụm', fontsize=10)
        axes[i].set_ylabel(feature, fontsize=10)
        axes[i].set_xticks(range(len(cluster_means)))
        axes[i].set_xticklabels([f'Cụm {j}' for j in range(len(cluster_means))])
        
        # Thêm giá trị
        for j, bar in enumerate(bars):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{height:.0f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    save_plot_safe('cluster_comparison.png', 'biểu đồ so sánh cụm')
    
except Exception as e:
    print(f"✗ Lỗi khi tạo biểu đồ so sánh: {e}")

# 5. Biểu đồ tròn
try:
    plt.figure(figsize=(10, 8))
    cluster_counts = df['Cluster'].value_counts().sort_index()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    plt.pie(cluster_counts.values, 
           labels=[f'Cụm {i}\n({count} khách hàng)' for i, count in cluster_counts.items()],
           autopct='%1.1f%%', colors=colors[:len(cluster_counts)], startangle=90)
    plt.title('Phân bố số lượng khách hàng theo cụm', fontsize=14)
    save_plot_safe('cluster_distribution.png', 'biểu đồ tròn phân bố')
    
except Exception as e:
    print(f"✗ Lỗi khi tạo biểu đồ tròn: {e}")

# 6. Radar chart
try:
    cluster_means = df.groupby('Cluster')[features].mean()
    cluster_means_normalized = cluster_means.copy()
    for feature in features:
        min_val = cluster_means[feature].min()
        max_val = cluster_means[feature].max()
        if max_val > min_val:
            cluster_means_normalized[feature] = (cluster_means[feature] - min_val) / (max_val - min_val)
        else:
            cluster_means_normalized[feature] = 0.5
    
    angles = [n / float(len(features)) * 2 * pi for n in range(len(features))]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, cluster in enumerate(cluster_means_normalized.index):
        values = cluster_means_normalized.loc[cluster].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, 
               label=f'Cụm {cluster}', color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_ylim(0, 1)
    ax.set_title('Đặc điểm các cụm khách hàng (Radar Chart)', size=14, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    save_plot_safe('cluster_radar.png', 'biểu đồ radar')
    
except Exception as e:
    print(f"✗ Lỗi khi tạo biểu đồ radar: {e}")

# 7. Heatmap correlation
try:
    plt.figure(figsize=(10, 8))
    df_corr = df[features + ['Cluster']].corr()
    sns.heatmap(df_corr, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, fmt='.2f')
    plt.title('Ma trận tương quan giữa các đặc trưng và cụm', fontsize=14)
    plt.tight_layout()
    save_plot_safe('correlation_heatmap.png', 'heatmap tương quan')
    
except Exception as e:
    print(f"✗ Lỗi khi tạo heatmap: {e}")

# 8. Box plots
try:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        sns.boxplot(x='Cluster', y=feature, data=df, ax=axes[i], palette='viridis')
        axes[i].set_title(f'Phân bố {feature} theo cụm', fontsize=12)
        axes[i].set_xlabel('Cụm', fontsize=10)
        axes[i].set_ylabel(feature, fontsize=10)
    
    plt.tight_layout()
    save_plot_safe('cluster_boxplots.png', 'box plots')
    
except Exception as e:
    print(f"✗ Lỗi khi tạo box plots: {e}")

# ========== LUU KẾT QUẢ ==========
print("\n=== BƯỚC 7: LƯU KẾT QUẢ ===")

# Lưu file Excel
try:
    excel_output = os.path.join(output_dir, 'khach_hang_clustered.xlsx')
    df.to_excel(excel_output, index=False)
    print(f"✓ Đã lưu kết quả phân cụm: {excel_output}")
except Exception as e:
    print(f"✗ Lỗi khi lưu file Excel: {e}")

# Tạo báo cáo tổng hợp
try:
    report_file = os.path.join(output_dir, 'cluster_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=== BÁO CÁO PHÂN CỤM KHÁCH HÀNG ===\n\n")
        f.write(f"Số lượng khách hàng: {len(df)}\n")
        f.write(f"Số cụm: {optimal_k}\n")
        f.write(f"Các đặc trưng sử dụng: {', '.join(features)}\n\n")
        
        f.write("=== THỐNG KÊ CÁC CỤM ===\n")
        for cluster in range(optimal_k):
            cluster_data = df[df['Cluster'] == cluster]
            f.write(f"\nCụm {cluster} ({len(cluster_data)} khách hàng):\n")
            for feature in features:
                mean_val = cluster_data[feature].mean()
                f.write(f"  {feature}: {mean_val:.2f}\n")
    
    print(f"✓ Đã tạo báo cáo tổng hợp: {report_file}")
except Exception as e:
    print(f"✗ Lỗi khi tạo báo cáo: {e}")

# ========== KẾT LUẬN ==========
print("\n=== KẾT LUẬN PHÂN TÍCH ===")

# Đặt tên cho các cụm dựa trên đặc điểm
cluster_names = {}
for cluster in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster]
    avg_age = cluster_data['Tuổi'].mean()
    avg_income = cluster_data['Thu nhập'].mean()
    avg_spending = cluster_data['Tổng chi tiêu'].mean()
    
    if avg_income > df['Thu nhập'].mean() and avg_spending > df['Tổng chi tiêu'].mean():
        cluster_names[cluster] = "Khách hàng VIP - Thu nhập cao, chi tiêu nhiều"
    elif avg_age > df['Tuổi'].mean():
        cluster_names[cluster] = "Khách hàng lớn tuổi - Mua sắm ít"
    else:
        cluster_names[cluster] = "Khách hàng trẻ - Thu nhập thấp"

for cluster, name in cluster_names.items():
    cluster_data = df[df['Cluster'] == cluster]
    print(f"\n{name}:")
    print(f"  Số lượng: {len(cluster_data)} khách hàng")
    print(f"  Tuổi trung bình: {cluster_data['Tuổi'].mean():.1f}")
    print(f"  Thu nhập trung bình: ${cluster_data['Thu nhập'].mean():.0f}")
    print(f"  Mua hàng trung bình: {cluster_data['Số lần mua hàng/năm'].mean():.1f} lần/năm")
    print(f"  Chi tiêu trung bình: ${cluster_data['Tổng chi tiêu'].mean():.0f}")

# Liệt kê các file đã tạo
print(f"\n=== CÁC FILE ĐÃ TẠO TRONG {output_dir} ===")
try:
    files = os.listdir(output_dir)
    png_files = [f for f in files if f.endswith('.png')]
    other_files = [f for f in files if not f.endswith('.png')]
    
    print("Biểu đồ:")
    for i, file in enumerate(sorted(png_files), 1):
        print(f"  {i}. {file}")
    
    print("\nFile khác:")
    for file in sorted(other_files):
        print(f"  - {file}")
        
except Exception as e:
    print(f"✗ Lỗi khi liệt kê files: {e}")

print("\n=== HOÀN THÀNH PHÂN TÍCH ===")
print("Tất cả biểu đồ và báo cáo đã được lưu thành công!")


