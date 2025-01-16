import torch
import tomesd
from DeepCache import DeepCacheSDHelper
from diffusers import StableDiffusionControlNetPipeline, UniPCMultistepScheduler, ControlNetModel
import matplotlib.pyplot as plt
from imageio.v3 import imwrite, imread
from skimage.util import invert
import numpy as np

def callback_dynamic_cfg(_, step_index, timestep, callback_kwargs):
    return callback_kwargs

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", 
                                             torch_dtype=torch.float32)
pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", 
                                                        controlnet=controlnet,
                                                        safety_checker=None, 
                                                        use_safetensors=True,
                                                        torch_dtype=torch.float32).to('cuda')
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

helper = DeepCacheSDHelper(pipe=pipe)
helper.set_params(
    cache_interval=5,
    cache_branch_id=0,
)
helper.enable()


tomesd.apply_patch(pipe, ratio=0.5)
pipe.enable_xformers_memory_efficient_attention()

pipe.unet.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)

prompt = "ship, high contrast grayscale drawing, only contours, wide lines, white background"
negative_prompt = "many lines, bad anatomy, worst quality, bad quality"
img = imread('images/scribble.png')
img = invert(img)[..., :3]
img[img != 255] = 0
img[img == 255] = 1

with torch.inference_mode():
    images = pipe(prompt, [img], num_inference_steps=50, height=512, width=512, negative_prompt=negative_prompt, output_type='np').images[0] * 255
    
    imwrite('images/gen.png', images.astype('uint8'))
