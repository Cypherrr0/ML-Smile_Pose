IMG_SIZE = 100
import cv2
import dlib
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 加载人脸检测器和人脸关键点检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('D:\\CollegeStuff\\shape_predictor_68_face_landmarks.dat')
def load_img(img_dir):
    images = []
    # Read image
    for filename in os.listdir(img_dir):
        if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(img_dir, filename))
            if img is None:
                print(f"Image file {filename} not found.")
            else:
                # Resize image
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                images.append(img)
    return np.array(images)

# Data preprocessing
def preprocess_data(images):
    # Normalize the image data to the range 0-1.
    images = images.astype('float32') / 255.0
    return images

def extract_features(images):
    all_features = []
    # 读取图片
    for image in images:
        # 假设整个图片就是一个人脸
        height, width = image.shape
        face = dlib.rectangle(0, 0, width, height)
        # 提取68个特征点
        landmarks = predictor(image, face)
        # 提取五官特征点
        features = []
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            features.append((x, y))
        features = np.array(features).flatten()
        all_features.append(features)
    return all_features

images = load_img('files_choped')
# 初始化一个空的列表来存储特征向量
features = []
features = extract_features(images)
images = preprocess_data(images)

from sklearn.preprocessing import StandardScaler
def normalize_features(features):
    # 初始化StandardScaler对象
    scaler = StandardScaler()
    
    # 使用特征数据训练scaler，并对特征进行归一化
    normalized_features = scaler.fit_transform(features)
    
    return normalized_features
features = normalize_features(features)

# 使用KMeans进行聚类
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(features)

from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# 使用LOF进行异常检测
lof = LocalOutlierFactor(n_neighbors=300, contamination=0.003)
y_pred = lof.fit_predict(features)

# 将标记为-1的点视为异常点
outliers = np.where(y_pred == -1)[0]

print(f"Found {len(outliers)} outliers.")
print(f"The indices of the outliers are: {outliers}")

# 使用PCA将特征降到2维
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)

# 获取所有的簇标签
cluster_labels = kmeans.labels_
import matplotlib.image as mpimg
def get_image_by_index(index):
    # 打印图片
    img = mpimg.imread(f'files_choped/file{index+1}.jpg')
    return img

# 获取所有的簇标签
cluster_labels = kmeans.labels_

# 对于每个簇
for i in range(kmeans.n_clusters):
    # 绘制该簇的所有点
    plt.scatter(reduced_features[cluster_labels == i, 0], reduced_features[cluster_labels == i, 1], label=f'Cluster {i}')
# 使用特殊颜色标注异常点
plt.scatter(reduced_features[outliers, 0], reduced_features[outliers, 1], c='black', label='Outliers')
plt.legend()
plt.show()


# 对于每个簇
for i in range(kmeans.n_clusters):
    # 创建一个新的图形
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # 在第一个子图中，先绘制其他的簇
    for j in range(kmeans.n_clusters):
        if j != i:
            axs[0].scatter(reduced_features[cluster_labels == j, 0], reduced_features[cluster_labels == j, 1], color='white',
                            label=f'Cluster {j}')

    # 计算当前簇的中心
    cluster_center = np.mean(reduced_features[cluster_labels == i], axis=0)

    # 找到离中心最近和最远的点
    distances = np.linalg.norm(reduced_features[cluster_labels == i] - cluster_center, axis=1)
    typical_point_index = np.argmin(distances)
    atypical_point_index = np.argmax(distances)
    typical_point = reduced_features[typical_point_index]
    atypical_point = reduced_features[atypical_point_index]

    # 在第一个子图中，再绘制当前的簇和特殊的点
    axs[0].scatter(reduced_features[cluster_labels == i, 0], reduced_features[cluster_labels == i, 1], color='red',
                    label=f'Cluster {i}')
    axs[0].scatter(typical_point[0], typical_point[1], c='green', marker='x', label='Typical Point')
    axs[0].scatter(atypical_point[0], atypical_point[1], c='blue', marker='x', label='Atypical Point')

    # 在第二个和第三个子图中，展示最典型和最不典型点的图片（这需要你有一个函数可以根据索引获取对应的图片）
    typical_image = get_image_by_index(typical_point_index)  # 你需要自己实现这个函数
    atypical_image = get_image_by_index(atypical_point_index)  # 你需要自己实现这个函数
    axs[1].imshow(typical_image)
    axs[1].set_title('Typical Point')
    axs[2].imshow(atypical_image)
    axs[2].set_title('Atypical Point')

    # 在创建完所有的图形元素之后调用axs[0].legend()
    axs[0].legend()
    plt.show()



