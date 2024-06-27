IMG_SIZE = 100
import cv2
import dlib
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from math import radians
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
# Preprocessing
# Load Haar cascade classifier
face_cascade = cv2.CascadeClassifier('D:\\CollegeStuff\\haarcascade_frontalface_default.xml')
# 加载dlib的人脸检测器和人脸关键点检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('D:\\CollegeStuff\\shape_predictor_68_face_landmarks.dat')
# Read image path
path_read = "D:\\CollegeStuff\\大三上\\机器学习\\Mini Project\\Mini Project\\genki4k\\files"

"""
num = 1
print("开始检测...")
for file_name in os.listdir(path_read):
    aa = (path_read + "/" + file_name)

    img = cv2.imdecode(np.fromfile(aa, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    # Perform face detection using the cascade classifier
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # The path to store the generated single face image.
    path_save = "D:\\CollegeStuff\\大三上\\机器学习\\Mini Project\\Mini Project\\files_choped"

    if len(faces) == 0:
        # If no face is detected, resize the original image to 200x200 pixels and then copy it to a new folder.
        img_resized = cv2.resize(img, (200, 200), interpolation=cv2.INTER_CUBIC)
        cv2.imencode('.jpg', img_resized)[1].tofile(path_save + "\\" + "file" + str(num) + ".jpg")
        print(f"No face detected, original image size adjusted and copied, serial number:{num}")
    else:
        for (x, y, w, h) in faces:
            # Generate an empty image based on the size of the face
            img_blank = img[y:y + h, x:x + w]
            img_blank = cv2.resize(img_blank, (200, 200), interpolation=cv2.INTER_CUBIC)
        cv2.imencode('.jpg', img_blank)[1].tofile(path_save + "\\" + "file" + str(num) + ".jpg")
    num += 1
print("New images generated")
"""
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
                # Convert image to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
    
    return np.array(images)

# Read label
def load_labels(label_file):
    label_data = []
    with open(label_file, 'r') as file:
        for line in file.readlines():
            data = line.split()
            yaw, pitch, roll = map(float, map(float, data[1:]))
            label_data.append([yaw, pitch, roll])
    return np.array(label_data)

# Data preprocessing
def preprocess_data(images):
    # Normalize the image data to the range 0-1.
    images = images.astype('float32') / 255.0
    return images

import cv2
import numpy as np

def extract_features(image):
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 将图像的数据类型转换为np.uint8
    gray = gray.astype(np.uint8)
    
    # 将图像尺寸缩小
    gray = cv2.resize(gray, (128, 128))
    
    # 创建HOG对象
    hog = cv2.HOGDescriptor()
    
    # 使用HOG对象提取特征
    features = hog.compute(gray)
    
    # 将特征展平
    features = features.flatten()
    
    return features


images = load_img('files_choped')#files_choped
labels = load_labels('genki4k/labels.txt')
images = preprocess_data(images)
print("Start extracting")
features = [extract_features(image) for image in images]
print("Extract finished")
# Remove rows with None values
features = [f for f in features if f is not None]

features = np.array(features)



print("Start spliting")
# Split training and testing sets.
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.15, random_state=42)
# Split again into training and validation sets.
train_features, val_features, train_labels, val_labels = train_test_split(train_features, train_labels, test_size=0.25, random_state=2)
print("Spliting finished")
train_errors = []
val_errors = []
# 创建列表以收集训练损失和验证损失
train_losses = []
val_losses = []
# 构建模型
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
# 在每一轮迭代中，我们将训练模型，并收集训练误差和验证误差
print("Starting training")
for i in tqdm(range(1, 11)):
    model.set_params(n_estimators=i)
    model.fit(train_features, train_labels)
    train_predictions = model.predict(train_features)
    val_predictions = model.predict(val_features)
    train_errors.append(mean_absolute_error(train_labels, train_predictions))
    val_errors.append(mean_absolute_error(val_labels, val_predictions))
    train_error = mean_absolute_error(train_labels, train_predictions)
    val_error = mean_absolute_error(val_labels, val_predictions)
    train_losses.append(train_error)
    val_losses.append(val_error)
    # 打印训练误差和验证误差
    print("Training MSE: ", train_errors[-1])
    print("Validation MSE: ", val_errors[-1])

# 在所有轮次结束后，使用最优模型在测试集上进行预测
test_predictions = model.predict(test_features)
# 计算测试集的MSE和MAE
test_mse = mean_squared_error(test_labels, test_predictions)
test_mae = mean_absolute_error(test_labels, test_predictions)
# 输出测试集的MSE和MAE
print("Test MSE: ", test_mse)
print("Test MAE: ", test_mae)

# 选择一个测试样本
sample_index = np.random.randint(len(test_labels))  # 随机选择一个测试样本
sample_features = test_features[sample_index]  # 获取该样本的特征
sample_label = test_labels[sample_index]  # 获取该样本的真实标签

# 使用模型进行预测
sample_features_reshaped = sample_features.reshape(1, -1)  # 重塑特征以匹配模型的输入形状
sample_prediction = model.predict(sample_features_reshaped)  # 使用模型进行预测

# 打印真实标签和预测标签
print("True label: ", sample_label)
print("Predicted label: ", sample_prediction)

# 绘制训练误差和验证误差
plt.figure()
plt.plot(train_errors, label='Training error')
plt.plot(val_errors, label='Validation error')
plt.legend()
plt.title('Training and validation error')
plt.xlabel('Number of trees')
plt.ylabel('Mean Absolute Error')
plt.show()

# 绘制训练误差和验证误差
plt.figure()
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.xlabel('Number of trees')
plt.ylabel('Mean Absolute Error')
plt.show()