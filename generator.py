import numpy as np
from PIL import Image
from multiprocessing import Process, Queue
import gc

from skimage.util import invert

import torch
import tomesd
from DeepCache import DeepCacheSDHelper
from diffusers import (
    StableDiffusionControlNetPipeline, 
    UniPCMultistepScheduler, ControlNetModel
)

class Generator:
    def __init__(self, output_queue: Queue, progress_queue: Queue):
        self.output_queue = output_queue
        self.progress_queue = progress_queue
        
    def callback(self, _, step_index, timestep, callback_kwargs):
        self.progress_queue.put(('progressbar_step', None))
        return callback_kwargs
    
    def generate(self, img: np.ndarray, prompt: str, negative_prompt: str = 'many lines, bad anatomy, worst quality, bad quality'):
        # self.output_queue.put([Image.open('images/gen.png'), Image.open('images/gen.png'), Image.open('images/gen.png')])
        # return
        img = invert(img)
        img[img != 255] = 0 
        img[img == 255] = 1 
        
        print('Настройка Stable Diffusion...')
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", 
                                                    torch_dtype=torch.float32).to('cuda')
        pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", 
                                                                controlnet=controlnet,
                                                                safety_checker=None, 
                                                                use_safetensors=True,
                                                                torch_dtype=torch.float32).to('cuda')

        helper = DeepCacheSDHelper(pipe=pipe)
        helper.set_params(
            cache_interval=5,
            cache_branch_id=0,
        )
        helper.enable()

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        pipe.enable_xformers_memory_efficient_attention()

        pipe.unet.to(memory_format=torch.channels_last)
        pipe.vae.to(memory_format=torch.channels_last)
        
        tomesd.apply_patch(pipe)
        
        print('Stable Diffusion успешно настроен.')
        
        base_prompt = ', drawing, only contours, wide lines, white background, great quality'

        generated_images = pipe(
            prompt + base_prompt, [img], 
            num_inference_steps=50, 
            negative_prompt=negative_prompt,
            height=512, width=512, 
            num_images_per_prompt=3,
            callback_on_step_end=self.callback
        ).images
        
        del pipe, controlnet
        gc.collect()
        torch.cuda.empty_cache()
            
        self.output_queue.put(generated_images)
    
    def start_generation(self, image: np.ndarray, prompt: str, negative_prompt: str = ''):
        process = Process(target=self.generate, args=(image, prompt, negative_prompt))
        process.start()
        