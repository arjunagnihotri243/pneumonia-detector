import cv2
import tensorflow as tf

CATEGORIES = ["NORMAL", "PNEUMONIA"]

def prepare(file_path):
    IMG_SIZE = 100
    img_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (100,100))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("models/model.model")


prediction = model.predict([prepare('xray_dataset_covid19/test/PNEUMONIA/SARS-10.1148rg.242035193-g04mr34g0-Fig8c-day10.jpeg')])
print(CATEGORIES[int(prediction[0][0])])