# type: ignore
from ucimlrepo import fetch_ucirepo
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Tải bộ dữ liệu Iris
iris = fetch_ucirepo(id=53)
# Hiển thị thông tin về bộ dữ liệu
# Lấy dữ liệu đặc trưng và nhãn
X = iris.data.features      # DataFrame các đặc trưng
y = iris.data.targets       # DataFrame nhãn thực tế

# Chuẩn hóa dữ liệu đặc trưng
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✓ Đã chuẩn hóa dữ liệu đặc trưng. Ví dụ 5 dòng đầu:")
print(X_scaled[:5])

# Mã hóa nhãn
le = LabelEncoder()
y_encoded = le.fit_transform(y.values.ravel())
print("✓ Đã mã hóa nhãn. 10 nhãn đầu tiên:", y_encoded[:10])

# Tạo thư mục lưu output
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
iris_output_dir = os.path.join(project_root, 'irisoutput')
os.makedirs(iris_output_dir, exist_ok=True)

# Thống kê mô tả các đặc trưng
print(X.describe())
print(X.isnull().sum())

# Trực quan hóa phân phối từng đặc trưng và lưu file
features = X.columns
for feature in features:
    plt.figure(figsize=(6, 4))
    sns.histplot(X[feature], kde=True, bins=20)
    plt.title(f'Phân phối {feature}')
    plt.xlabel(feature)
    plt.ylabel('Số lượng mẫu')
    plt.tight_layout()
    file_path = os.path.join(iris_output_dir, f'iris_hist_{feature.replace(" ", "_")}.png')
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Đã lưu biểu đồ histogram: {file_path}")

# Boxplot so sánh đặc trưng theo nhãn và lưu file
df_iris = X.copy()
df_iris['class'] = y.values.ravel()
# hoặc
# df_iris['class'] = y.values.squeeze()
for feature in features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='class', y=feature, data=df_iris)
    plt.title(f'Boxplot {feature} theo nhãn')
    plt.xlabel('Loài hoa')
    plt.ylabel(feature)
    plt.tight_layout()
    file_path = os.path.join(iris_output_dir, f'iris_boxplot_{feature.replace(" ", "_")}.png')
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Đã lưu biểu đồ boxplot: {file_path}")

