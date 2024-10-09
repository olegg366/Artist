# Импортируем необходимые библиотеки
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from sklearn.model_selection import train_test_split
import numpy as np

# Путь к сохраненной модели
model_path = 'mlmodels/static'

# Загружаем данные и разделяем их на обучающую и тестовую выборки
# x_train, x_test - входные данные (features)
# y_train, y_test - выходные данные (labels)
x_train, x_test, y_train, y_test = train_test_split(
    np.load('dataset/static_X.npy'),  # Загружаем входные данные
    np.load('dataset/static_Y.npy'),  # Загружаем выходные данные
    test_size=0.3,                   # Размер тестовой выборки (30%)
    random_state=42                  # Фиксируем случайный seed для воспроизводимости
)

# Создаем конвертер для преобразования модели в формат TensorRT
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=model_path,  # Путь к сохраненной модели
    precision_mode=trt.TrtPrecisionMode.FP32  # Указываем режим точности (FP32)
)

# Максимальный размер батча для оптимизации
MAX_BATCH_SIZE = 128

# Функция для генерации входных данных для оптимизации
def input_fn():
    batch_size = MAX_BATCH_SIZE  # Размер батча
    x = x_test[0:batch_size]     # Берем первые batch_size элементов из тестовой выборки
    yield [x]                    # Возвращаем батч в виде списка

# Преобразуем модель в формат TensorRT
trt_func = converter.convert()

# Строим оптимизированную модель с использованием input_fn
converter.build(input_fn=input_fn)

# Путь для сохранения оптимизированной модели
OUTPUT_SAVED_MODEL_DIR = 'mlmodels/static_tftrt'

# Сохраняем оптимизированную модель
converter.save(output_saved_model_dir=OUTPUT_SAVED_MODEL_DIR)