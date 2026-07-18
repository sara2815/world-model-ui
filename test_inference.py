import torch
from diffusers import Flux2KleinPipeline

device = "cuda"
dtype = torch.bfloat16

pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-base-4B", torch_dtype=dtype)
# pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU

prompt = "A cat holding a sign that says hello world"

pipe = pipe.to("cuda")

generator = torch.Generator(device="cuda").manual_seed(0)

image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=4.0,
    num_inference_steps=50,
    generator=generator,
).images[0]

image.save("flux-klein.png")