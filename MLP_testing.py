import keras
import numpy as np
import tensorflow as tf
def mlp_digits_predict(model, image_file):
   image_size = 28
   img = keras.preprocessing.image.load_img(image_file, target_size=(image_size, image_size), color_mode='grayscale')
   img_arr = np.expand_dims(img, axis=0)
   img_arr = 1 - img_arr/255.0
   img_arr = img_arr.reshape((1, image_size*image_size))
   result = (model.predict([img_arr]) > 0.5).astype("int32")
   return result[0]

model = tf.keras.models.load_model('models/MLP.h5')
print(mlp_digits_predict(model, 'data/1.png'), "правильный ответ - 1")
print(mlp_digits_predict(model, 'data/5.png'), "правильный ответ - 5")
print(mlp_digits_predict(model, 'data/9.png'), "правильный ответ - 9")
print(mlp_digits_predict(model, 'data/2.png'), "правильный ответ - 2")
print(mlp_digits_predict(model, 'data/6.png'), "правильный ответ - 6")