import torch
import tomesd
from DeepCache import DeepCacheSDHelper
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, UniPCMultistepScheduler, ControlNetModel

def callback_dynamic_cfg(_, step_index, timestep, callback_kwargs):
        return callback_kwargs

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", 
                                             torch_dtype=torch.float32)
pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", 
                                                        controlnet=controlnet,
                                                        safety_checker=None, 
                                                        use_safetensors=True,
                                                        torch_dtype=torch.float32)
tomesd.apply_patch(pipe, ratio=0.5)

helper = DeepCacheSDHelper(pipe)
helper.set_params(
    cache_interval=5,
    cache_branch_id=0,
)
helper.enable()

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

pipe.enable_sequential_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

pipe.unet.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)

prompt = "house, single color, child's drawing"
negative_prompt = "many lines"
img = Image.open('images/scribble.png')

generator = torch.manual_seed(2023)

with torch.inference_mode():
    image = pipe(prompt, img, num_inference_steps=50, height=512, width=512, negative_prompt=negative_prompt, generator=generator).images[0]

    image.save('images/gen.png')
