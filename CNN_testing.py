import tensorflow as tf
import numpy as np
def test(image,model):
    predictions = model.predict(image)
    if predictions[0] >= 0.5:
        print("Картинка - собака")
    else:
        print("Картинка - кошка")
def img_convert(img_path):
    test_image = tf.keras.preprocessing.image.load_img(img_path, target_size=(150,150))
    test_image = tf.keras.preprocessing.image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image / 255.0
    return test_image

model = tf.keras.models.load_model("models/CNN.h5")
dog_img = img_convert("data/dog_test.png")
cat_img = img_convert("data/cat_test.png")
dog2_img = img_convert("data/test_dog2.jpg")
test(dog_img, model)
test(cat_img, model)
test(dog2_img, model)

