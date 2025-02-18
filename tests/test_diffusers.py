# Импорт необходимых библиотек
import torch
import tomesd
from DeepCache import DeepCacheSDHelper
from diffusers import (
    StableDiffusionControlNetPipeline, 
    UniPCMultistepScheduler, ControlNetModel
)
import matplotlib.pyplot as plt
from imageio.v3 import imwrite, imread
from skimage.util import invert
import numpy as np

# Загрузка предобученной модели ControlNet для работы с контурными изображениями
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-scribble", 
    torch_dtype=torch.float32
)

# Создание пайплайна Stable Diffusion с ControlNet
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    controlnet=controlnet,
    safety_checker=None, 
    use_safetensors=True,
    torch_dtype=torch.float32
).to('cuda')

# Настройка планировщика для пайплайна
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Инициализация DeepCache для оптимизации работы пайплайна
helper = DeepCacheSDHelper(pipe=pipe)
helper.set_params(
    cache_interval=5,  # Интервал кэширования
    cache_branch_id=0,  # ID ветки кэширования
)
helper.enable()  # Включение DeepCache

# Применение патча для оптимизации памяти с использованием tomesd
tomesd.apply_patch(pipe, ratio=0.5)

# Включение эффективного использования памяти с помощью xformers
pipe.enable_xformers_memory_efficient_attention()

# Преобразование формата памяти UNet и VAE для повышения производительности
pipe.unet.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)

# Определение текстового запроса и негативного запроса
prompt = "flower, drawing, only contours, wide lines, white background"
negative_prompt = "many lines, bad anatomy, worst quality, bad quality"

# Загрузка и предобработка изображения с контурами
img = imread('images/scribble.png')
img = invert(img)  # Инвертирование цветов изображения
img[img != 255] = 0  # Установка всех пикселей, кроме белых, в 0
img[img == 255] = 1  # Установка белых пикселей в 1

# Генерация изображения с использованием пайплайна
with torch.inference_mode():
    images = pipe(
        prompt, 
        [img], 
        num_inference_steps=50, 
        height=512, width=512, 
        negative_prompt=negative_prompt, 
        num_images_per_prompt=3,
        output_type='np'
    ).images[0] * 255
    
    # Сохранение сгенерированного изображения
    imwrite('images/gen.png', images.astype('uint8'))