from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os


def get_the_name_of_class(_prediction) -> str:
    """Функция для вывода имени класса объекта"""
    classes_file_path: str = "../data/classes"
    data: list = open(classes_file_path).read().split("\n")

    return data[np.argmax(_prediction)]


def load_image(photo_path, show=False):
    """Функция для загрузки изображения с определёнными параметрами"""
    img = image.load_img(photo_path, target_size=(250, 250))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)  # (1, height, width, channels), add a dimension because the model
                                                     # expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
    # Для отображения фото
    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor


if __name__ == "__main__":
    # Папка с файлами для тестирования
    test_path: str = "../data/test/"
    files: list = os.listdir(test_path)
    # Загрузка модели
    model = load_model("../model/Complete/MyModel.keras")
    for file in files:
        # print(file)
        try:
            img_path = test_path + file
            new_image = load_image(img_path, True)

            # функция для классификации
            prediction = model.predict(new_image)
            # Сумма всех соотношений
            summary = sum(prediction[0])
            # print(prediction)
            calc_arr = np.vectorize(lambda t: round(t/summary, 4)*100)
            print(calc_arr(prediction))
            print(f"{file}: It is", get_the_name_of_class(prediction))
        except Exception as error:
            print(error)
