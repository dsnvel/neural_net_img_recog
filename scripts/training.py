from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


image_width, image_height = 250, 250  # Указываем разрешение для изображений к единому формату

directory_data_train = '../data/train'  # Путь до тренировочного датасета
directory_data_validation = '../data/validation'  # Указываем путь к проверочной выборке validation_data_dir

# Записываем все имеющиеся классы в директории с тренировчным датасетом в файл classes
classes_data: list = os.listdir(directory_data_train)
with open("../data/classes", "w") as classes_file:
    for folder in classes_data:
        print(folder)
        classes_file.write(folder + "\n")

# Устанавливаем необходимые параметры
train_sample = 200
validation_sample = 200
epochs = 40
lot_size = 50  # batch_size
if K.image_data_format() != 'channels_first':
    input_shape = (image_width, image_height, 3)
else:
    input_shape = (3, image_width, image_height)

pattern = Sequential()  # Создание модели

# Первый слой нейросети
pattern.add(Conv2D(32, (3, 3), input_shape=input_shape))
pattern.add(Activation('relu'))
pattern.add(MaxPooling2D(pool_size=(2, 2)))

# Второй слой нейросети
pattern.add(Conv2D(32, (3, 3)))
pattern.add(Activation('relu'))
pattern.add(MaxPooling2D(pool_size=(2, 2)))

# Третий слой нейросети
pattern.add(Conv2D(64, (3, 3)))
pattern.add(Activation('relu'))
pattern.add(MaxPooling2D(pool_size=(2, 2)))

# Aктивация, свертка, объединение, исключение
pattern.add(Flatten())
pattern.add(Dense(64))
pattern.add(Activation('relu'))
pattern.add(Dropout(0.5))
pattern.add(Dense(14))  # число классов
pattern.add(Activation('softmax'))

# Cкомпилируем модель с выбранными параметрами. Также укажем метрику для оценки.
pattern.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

# Задаём параметры аугментации, подстраивает изображения под определённые значения, формирует новые фото
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # коэффициент масштабирования
    shear_range=0.2,  # Интенсивность сдвига
    zoom_range=0.2,  # Диапазон случайного увеличения
    horizontal_flip=True)  # Произвольный поворот по горизонтали
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Предобработка обучающей выборки
train_processing = train_datagen.flow_from_directory(
    directory_data_train,
    target_size=(image_width, image_height),  # Размер изображений
    batch_size=lot_size,  # Размер пакетов данных
    class_mode='categorical')  # Режим класса

# Предобработка тестовой выборки
validation_processing = test_datagen.flow_from_directory(
    directory_data_validation,
    target_size=(image_width, image_height),
    batch_size=lot_size,
    class_mode='categorical')

# Генерируем процесс обучения модели
pattern.fit_generator(
    train_processing,  # Помещаем обучающую выборку
    steps_per_epoch=train_sample // lot_size,
    # количество итераций пакета до того, как период обучения считается завершенным
    epochs=epochs,  # Указываем количество эпох
    validation_data=validation_processing,  # Помещаем проверочную выборку
    validation_steps=validation_sample // lot_size)  # Количество итерации, но на проверочном пакете данных

pattern.save_weights('../weights/ForTest/first_model_weights.h5')  # Сохранение весов модели
pattern.save('../model/ForTest/MyModel.keras')  # Сохранение модели
