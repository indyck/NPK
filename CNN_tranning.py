import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
EPOCHS = 100
# Создание модели
model = Sequential()

# Добавление слоев свертки и пулинга
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

# Преобразование 3D-представления в 1D-представление
model.add(Flatten())

# Полносвязные слои для классификации
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Предварительная обработка и загрузка данных
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
train_generator = train_datagen.flow_from_directory(
    'data/datasets/cats_and_dogs_filtered',
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

model.fit(train_generator,steps_per_epoch=100,epochs=EPOCHS)
model.save("models/CNN.h5")