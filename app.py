from stable_diffusion_videos import StableDiffusionWalkPipeline, Interface
import torch

pipeline = StableDiffusionWalkPipeline.from_pretrained(
	# 'CompVis/stable-diffusion-v1-4',
	'runwayml/stable-diffusion-v1-5',
	torch_dtype=torch.float16,
	revision='fp16',
).to('cuda')

interface = Interface(pipeline)
interface.launch()
