import torch
import tomesd
from DeepCache import DeepCacheSDHelper
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, UniPCMultistepScheduler, ControlNetModel
import matplotlib.pyplot as plt

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

prompt = "house, single color, child's drawing"
negative_prompt = "many lines"
img = Image.open('images/scribble.png')

with torch.inference_mode():
    images = pipe(prompt, img, num_inference_steps=50, height=512, width=512, negative_prompt=negative_prompt, num_images_per_prompt=3, output_type='np').images

    for img in images:
        plt.imshow(img)
        plt.show()
