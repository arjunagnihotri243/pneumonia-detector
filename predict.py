import cv2
import tensorflow as tf

CATEGORIES = ["NORMAL", "PNEUMONIA"]

def prepare(file_path):
    IMG_SIZE = 100
    img_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (100,100))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("models/model.model")

filePath = input("Enter File Path of Image you want to predict: ")
prediction = model.predict([prepare(filePath)])
print(CATEGORIES[int(prediction[0][0])])