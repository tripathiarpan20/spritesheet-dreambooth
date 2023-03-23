
import base64
import io
from PIL import Image
import numpy as np
import cv2

import torch
from torch import autocast
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionPipeline, StableDiffusionControlNetPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler, StableDiffusionDepth2ImgPipeline, UniPCMultistepScheduler
from diffusers.utils import load_image
from pathlib import Path
import openai
from controlnet_aux import HEDdetector
from diffusers.utils import load_image

def init_hed_controlnet(local_model_path = "./spritesheet_dreambooth_merge_kaliyuga_extract"):
  controlnet = ControlNetModel.from_pretrained(
      "fusing/stable-diffusion-v1-5-controlnet-hed", torch_dtype=torch.float16
  )

  pipe = StableDiffusionControlNetPipeline.from_pretrained(
      local_model_path, controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
  ).to('cuda')

  pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
  return pipe


#'image_path' is a local path to the image
def load_image(image_path):
  init_img = Image.open(image_path).convert("RGB").resize((512, 512))
  #returns a PIL Image
  return init_img


#'image_path' is a local path to the image or bytearray or bytestream
#reference: https://github.com/pytorch/serve/blob/master/ts/torch_handler/vision_handler.py
def load_image_generalised(image_path, resize = False):
  path = Path(image_path)

  print('Loading file at => ', path)

  init_img = None
  if isinstance(image_path, str) or path.is_file():
      init_img = Image.open(image_path).convert("RGB")
  else:
    # if the image is a string of bytesarray.
    init_img = base64.b64decode(image_path)

  # If the image is sent as bytesarray
  if isinstance(image_path, (bytearray, bytes)):
      init_img = Image.open(io.BytesIO(image_path))
      init_img = init_img.convert("RGB")

  
  #returns a PIL Image
  if resize:
    return init_img.resize((512,512))
  else:
    return init_img


def inference(pipe, \
              hed_image,\
              prompts = ["icy werewolf", "flame knight"], \
              num_inference_steps: int = 20,\
              guidance_scale: float =20,
              negative_pmpt:str = None,
              req_type = "asset",
              device = "cuda",
              seed = None):
  
  # print(prompts)
  prompts_postproc = None
  images = None
  if req_type == 'asset':
    #for `diffusers_summerstay_strdwvlly_asset_v2` model
    # prompts_postproc = [f'{prompt}, surrounded by completely black, strdwvlly style, completely black background, HD, detailed' for prompt in prompts]
    # negative_pmpt = "isometric, interior, island, farm, monochrome, glowing, text, character, sky, UI, pixelated, blurry"

    #for `stable-diffusion-2-depth` model
    prompts_postproc = [f'{prompt} spritesheettttt, right alignment, with a completely light green background' for prompt in zip(prompts)]

    if negative_pmpt is not None:  
      negative_prompt = [negative_pmpt for x in range(len(prompts_postproc))]
    else:
      negative_prompt = None
    # print(prompts_postproc[0], '!!!!!!!!!!\n', prompts_postproc[1])

    generator = None
    if seed is not None:
      generator = torch.Generator(device=device).manual_seed(seed)

    with autocast("cuda"):
        images = pipe(prompt=prompts_postproc,\
                    hed_image,
                    negative_prompt = negative_prompt,\
                    num_inference_steps = num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator = generator)
    images = images[0]

  #Returns a List of PIL Images
  return images

def inference_with_edge_guidance(canny_controlnet_pipe, init_image, prompts, negative_pmpt, canny_lower, canny_upper, num_inference_steps = 20):
    # This uses the edges from an init image to guide the generation of a new image.
    # it outputs an image in the standard diffusers format
    # The init image is an image whose outline and major shapes you want preserved in the output
    # Canny_lower and Canny_upper are thresholds on which edges will be kept. 100 for lower and 200 for upper is a good starting point for experimentation. They can go from 1 to 255, I think.
    
    if negative_pmpt is not None:  
      negative_prompt = [negative_pmpt for x in range(len(prompts))]
    else:
      negative_prompt = None

    #Converting PIL Image to OpenCV Image
    init_image = cv2.cvtColor(np.array(init_image), cv2.COLOR_RGB2BGR)
    edge_image = cv2.Canny(init_image,canny_lower,canny_upper)
    
    image = canny_controlnet_pipe(prompt=prompts, negative_prompt = negative_prompt, controlnet_hint=edge_image, num_inference_steps = num_inference_steps).images[0]

    return image


def inference_w_gpt(pipe, \
              init_img,\
              prompts = ["blue house", "blacksmith workshop"], \
              strength: float = 0.90,\
              num_inference_steps: int = 20,\
              guidance_scale: float =20,
              negative_pmpt:str = "base, ground, terrain, child's drawing, sillhouette, dark, shadowed, green blob, cast shadow on the ground, background pattern",
              req_type = "asset",
              device = "cuda",
              seed = 1024):
  
  # print(prompts)
  images = None
  if req_type == 'asset':

    images = []
    #for summerstay's magenta model
    adjs = [x.split()[0] for x in prompts]
    adjectives = [f"{adj} world" for adj in adjs]

    for idx in range(len(prompts)):
      prompt = """In creating art for video games, it is important that everything contributes to an overall style. If the style is 'candy world', then everything should be made of candy:
      * tree: gumdrop fruit and licorice bark
      * flower: lollipops with leaves
      For an 'ancient Japan' setting, the items are simply a variation of the items that might be found in ancient Japan. Some might be unchanged:
      * church: a Shinto shrine
      * tree: a gnarled, beautiful cherry tree that looks like a bonsai tree
      * tree stump: tree stump
      * stone: a stone resembling those in zen gardens
      If the style instead is '""" + adjectives[idx] + """' then the items might be:
    * """ + prompts[idx] + """:"""
      outtext = openai.Completion.create(
          model="davinci",
          prompt=prompt,
                max_tokens=256,
          temperature=0.5,
          stop=['\n','.']
          )
      response = outtext.choices[0].text
      print(prompt, '\n--------------------\n')
      print(response, '\n--------------------')

      prompts_postproc = "robust, thick trunk with visible roots, concept art of " + response + ", " + adjectives[idx] + ", game asset surrounded by pure magenta, view from above, studio ghibli and disney style, completely flat magenta background" 

      generator = None
      if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

      with autocast("cuda"):
        image = pipe(prompt=prompts_postproc,\
                    negative_prompt = negative_pmpt,\
                    image=init_img, 
                    strength=strength, 
                    num_inference_steps = num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator = generator)[0][0]
        images.append(image)
  
      
  #Returns a List of PIL Images
  return images