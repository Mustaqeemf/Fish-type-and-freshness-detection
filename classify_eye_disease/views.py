import os
import numpy as np
import tensorflow as tf
# from tensorflow.keras.models import Sequential,Model
# from tensorflow.keras.applications import MobileNetV2,VGG16
# from tensorflow.keras.layers import Conv2D,Flatten,Dropout,BatchNormalization,Dense
# from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
# from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,array_to_img,img_to_array
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from rest_framework.decorators import api_view
from rest_framework.response import Response


app_path = os.path.join(settings.BASE_DIR, 'classify_eye_disease')
classify_model_path = os.path.join(app_path, 'fish-classify.h5')
freshness_model_path = os.path.join(app_path, 'fish-freshness.h5')

classify_classes = {
    0: 'Black Sea Sprat',
    1: 'Gilt-Head Bream',
    2: 'Hourse Mackerel',
    3: 'Red Mullet',
    4: 'Red Sea Bream',
    5: 'Sea Bass',
    6: 'Shrimp',
    7: 'Striped Red Mullet',
    8: 'Trout'
}


def preprocess_image_classify(image_path, target_size=(224, 224)):
    try:
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image
    except FileNotFoundError:
        print("File not found at path:", image_path)
        return None
    except Exception as e:
        print("An error occurred while preprocessing the image:", e)
        return None


def preprocess_image_freshness(image_path):
    try:
        image = tf.keras.preprocessing.image.load_img(image_path, color_mode='rgb', target_size=(50, 50))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image / 255.0
        return image
    except FileNotFoundError:
        print("File not found at path:", image_path)
        return None
    except Exception as e:
        print("An error occurred while preprocessing the image:", e)
        return None


def load_classify_model():
    return tf.keras.models.load_model(classify_model_path)


def load_freshness_model():
    return tf.keras.models.load_model(freshness_model_path)


def predict_single_img_classify(img_path):
    classify_model_load = load_classify_model()
    test_image = preprocess_image_classify(img_path)
    if test_image:
        classify_predictions = classify_model_load.predict(test_image)
        predicted_classify_index = np.argmax(classify_predictions, axis=1)[0]
        return classify_classes[predicted_classify_index]
    else:
        return 'None'


def predict_single_img_freshness(image_path, classes):
    freshness_model_load = load_freshness_model()
    test_image = preprocess_image_freshness(image_path)
    if test_image:
        test_image = test_image.reshape((1,) + test_image.shape)  # Reshape to match the input shape of the model
        prediction = freshness_model_load.predict(test_image)
        predicted_class_index = prediction.argmax(axis=-1)[0]
        predicted_class = classes[predicted_class_index]
        return True, predicted_class
    else:
        return 'None'


@api_view(['GET'])
def index(request):
    fish_category = request.FILES['fish_category']
    fish_freshness = request.FILES['fish_freshness']
    fss = FileSystemStorage()
    file = fss.save(fish_category.name, fish_category)
    file1 = fss.save(fish_freshness.name, fish_freshness)
    file_url = fss.url(file)
    file_url_1 = fss.url(file1)
    complete_path_classify = settings.MEDIA_FILE_PATH + file_url
    complete_path_freshness = settings.MEDIA_FILE_PATH + file_url_1
    img_result_classify = predict_single_img_classify(complete_path_classify)
    img_result_freshness = predict_single_img_freshness(complete_path_freshness)
    response_dict = {
        'fish_name': img_result_classify,
        'freshness': img_result_freshness,
    }
    return Response(response_dict)
