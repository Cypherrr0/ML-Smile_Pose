IMG_SIZE = 100
import cv2
import os
import numpy as np
import shutil
import tensorflow
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# Preprocessing
# Load Haar cascade classifier
face_cascade = cv2.CascadeClassifier('D:\\CollegeStuff\\haarcascade_frontalface_default.xml')

# Read image path
path_read = "D:\\CollegeStuff\\大三上\\机器学习\\Mini Project\\Mini Project\\genki4k\\files"
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


# Load dataset
def load_img(img_dir):
    images = []
    # Read image
    for filename in os.listdir(img_dir):
        if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(img_dir, filename))
            if img is None:
                print(f"Image file {filename} not found.")
            else:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)

    return np.array(images)

def load_labels(label_file):
    labels = []
    # Read label
    with open(label_file, 'r') as f:
        for line in f:
            labels.append(int(line.strip()[0]))
    labels = np.array(labels)
    return labels



# Data preprocessing
def preprocess_data(images):
    # Normalize the image data to the range 0-1.
    images = images.astype('float32') / 255.0
    return images

images = load_img('genki4k/files')
labels = load_labels('genki4k/labels.txt')
images = preprocess_data(images)
# Split training and testing sets.
train_data, test_data, train_labels, test_labels = train_test_split(images, labels, test_size=0.15, random_state=42)

# Split again into training and validation sets.
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.25, random_state=2)


from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.summary()

optimizer = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Use data augmentation.
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)
print("start training")
# Start training
history = model.fit(datagen.flow(train_data, train_labels, batch_size=32),
                    steps_per_epoch=len(train_data) / 32,
                    epochs=20,
                    validation_data=(val_data, val_labels),
                    verbose=1)


loss, accuracy = model.evaluate(test_data, test_labels)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

# Visualize training result
plt.figure(figsize=(12,5))
# Plot accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'],label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()



# Predict the values from the validation dataset
y_pred = model.predict(test_data)
# Convert predictions to binary classes
y_pred_classes = (y_pred > 0.5).astype(int).reshape(-1)
# Convert validation observations to one hot vectors
y_true = test_labels
# Compute the confusion matrix
confusion_mtx = confusion_matrix(y_true, y_pred_classes) 

# Plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


