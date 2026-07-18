import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("SG161222/Realistic_Vision_V5.1_noVAE", dtype=torch.bfloat16, device_map="cuda")

prompt = "a blue sky with few clouds"
image = pipe(prompt).images[0]
image.save("blue_sky.png")