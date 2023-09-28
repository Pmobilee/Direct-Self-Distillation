import os
import sys
import torch
import importlib
import tqdm
import numpy as np
from omegaconf import OmegaConf
import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import *
import torch.nn as nn
# import matplotlib.pyplot as plt
import copy
import wandb
import math
import traceback
from pytorch_fid import fid_score
import shutil
from self_distillation import *
from distillation import *
from saving_loading import *
from util import *
import matplotlib.pyplot as plt



@torch.no_grad()
def generate_images(model, sampler, num_imgs=1, steps=20, eta=0.0, scale=3.0, x_T=None, class_prompt=None, keep_intermediates=False,x_0=False):
    """
    Params: model, sampler, num_imgs=1, steps=20, eta=0.0, scale=3.0, x_T=None, class_prompt=None, keep_intermediates=False. 
    Task: returns final generated samples from the provided model and accompanying sampler. Unless the class prompt is specified,
    all generated images are of one of the random classes. Pred_x0 and samples_ddim are identical when the final denoising step is returned.
    """
    NUM_CLASSES = 1000

    if x_0:
        total_steps=64
        intermediate_step = 0
    else:
        total_steps=steps   
        intermediate_step = None

    sampler.make_schedule(ddim_num_steps=total_steps, ddim_eta=eta, verbose=False)

    if class_prompt == None:
        class_prompt = torch.randint(0, NUM_CLASSES, (num_imgs,))

    with torch.no_grad():
        # with model.ema_scope(): # uncomment for EMA, unimportant for now as this function is only used for evaluation
            uc = model.get_learned_conditioning(
                    {model.cond_stage_key: torch.tensor(num_imgs*[1000]).to(model.device)})
            xc = torch.tensor(num_imgs*[class_prompt])
            c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
            
            _, _, x_T_copy, pred_x0, _ = sampler.sample(S=steps,
                                            conditioning=c,
                                            batch_size=1,
                                            shape=[3, 64, 64],
                                            verbose=False,
                                            x_T=x_T,  
                                            unconditional_guidance_scale=scale,
                                            unconditional_conditioning=uc, 
                                            eta=eta,
                                            keep_intermediates=keep_intermediates,
                                            intermediate_step=intermediate_step,
                                            total_steps=total_steps,
                                            steps_per_sampling=steps)
          
                                    
    # # display as grid
    # x_samples_ddim = model.decode_first_stage(pred_x0)
    # x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
    # grid = rearrange(x_samples_ddim, 'b c h w -> (b) c h w')
    # grid = make_grid(grid, nrow=1)

    # # to image
    # grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    # image = Image.fromarray(grid.astype(np.uint8))

    # display as grid
    
    

    # to image
    

    sample = ((pred_x0 + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()

    grid = rearrange(sample, 'b c h w -> (b) c h w')
    grid = make_grid(grid, nrow=1)
    
    # grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    # image = Image.fromarray(grid.astype(np.uint8))

    # image = Image.fromarray(sample)

    # Convert the tensor 'sample' to a numpy array
    #sample_np = sample[0].cpu().numpy()  # Assuming 'sample' is a tensor with batch dimension
    sample_np = grid.cpu().numpy()
    # Convert the numpy array to a uint8 data type (assuming it's in the range [0, 255])
    sample_np = np.uint8(sample_np)

    # Create a PIL Image from the numpy array
    image = Image.fromarray(sample_np)

    # Display the image
    # image.show()

    return image, x_T_copy, class_prompt, pred_x0

@torch.no_grad()
def generate_images_celeb(model, sampler, num_imgs=1, steps=20, total_steps=64, eta=0.0, scale=3.0, x_T=None, class_prompt=None, keep_intermediates=False, x_0=False):
    """
    Params: model, sampler, num_imgs=1, steps=20, eta=0.0, scale=3.0, x_T=None, class_prompt=None, keep_intermediates=False. 
    Task: returns final generated samples from the provided model and accompanying sampler. Unless the class prompt is specified,
    all generated images are of one of the random classes.
    """
    NUM_CLASSES = 1000
    sampler.make_schedule(ddim_num_steps=steps, ddim_eta=eta, verbose=False)
    if x_0:
        total_steps=64
        intermediate_step = 0
    else:
        total_steps=steps 
        intermediate_step = None
    if class_prompt == None:
        class_prompt = torch.randint(0, NUM_CLASSES, (num_imgs,))
    
    with torch.no_grad():
        # with model.ema_scope(): # uncomment for EMA, unimportant for now as this function is only used for evaluation

            samples_ddim, _, x_T_copy, pred_x0, a_t, _ = sampler.sample(S=steps,
                                            
                                            batch_size=1,
                                            shape=[3, 64, 64],
                                            verbose=False,
                                            x_T=x_T,
                                            
                                            unconditional_guidance_scale=scale,
                                            eta=eta,
                                            keep_intermediates=keep_intermediates,
                                            intermediate_step=intermediate_step,
                                            total_steps=total_steps,
                                            steps_per_sampling=steps)
          
                                    
    # display as grid
    x_samples_ddim = model.decode_first_stage(pred_x0)
    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
    grid = rearrange(x_samples_ddim, 'b c h w -> (b) c h w')
    grid = make_grid(grid, nrow=1)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    image = Image.fromarray(grid.astype(np.uint8))

    return image, x_T_copy, class_prompt


@torch.no_grad()
def ablation(sampler, steps=8, shape=(2, 2), celeb=False, prompt_list=None, noise_list=None, model_type=None, prompts=None):
    """
    Disclaimer: Written in part by ChatGPT
    Takes as input a sampler and a list of steps. For each step, it generates n images with starting noise shared between steps and returns them in a grid.
    """
    from PIL import Image, ImageDraw
    image_list = []
    
    # Quick and dirty, but works
    if celeb:
        if noise_list == None:
            noise_list = []
            prompt_list = []
            for i in range(shape[0] * shape[1]):
                image, noise, prompt, _ = generate_images_celeb(sampler.model, sampler, steps=steps, eta=0.0, scale=3.0, x_T=None, class_prompt=None, keep_intermediates=False,x_0=False)
                image_list.append(image)
                noise_list.append(noise)
                
        else:
            for i in range(shape[0] * shape[1]):
                noise = noise_list[i]
                image, noise, prompt, _ = generate_images_celeb(sampler.model, sampler, steps=steps, eta=0.0, scale=3.0, x_T=noise, class_prompt=None, keep_intermediates=False,x_0=False)
                image_list.append(image)
    else:
        if noise_list == None:
            noise_list = []
            prompt_list = []
            for i in range(shape[0] * shape[1]):
                if prompts != None:
                    class_prompt = prompts[i]
                image, noise, prompt, _ = generate_images(sampler.model, sampler, steps=steps, eta=0.0, scale=3.0, x_T=None, class_prompt=class_prompt, keep_intermediates=False,x_0=False)
                image_list.append(image)
                noise_list.append(noise)
                prompt_list.append(prompt)
        else:
            for i in range(shape[0] * shape[1]):
                noise = noise_list[i]
                prompt = prompt_list[i]
                image, noise, prompt, _ = generate_images(sampler.model, sampler, steps=steps, eta=0.0, scale=3.0, x_T=noise, class_prompt=prompt, keep_intermediates=False,x_0=False)
                image_list.append(image)
    

    # Set the size of the output image
    image_width = 256
    image_height = 256

    # Set the number of images to display per row and column
    images_per_row = shape[0]
    images_per_col = shape[1]

    # Set the spacing between images
    horizontal_spacing = 10
    vertical_spacing = 10

    # Calculate the total width and height of the output image
    total_width = (image_width * images_per_row) + ((images_per_row - 1) * horizontal_spacing)
    total_height = (image_height * images_per_col) + ((images_per_col - 1) * vertical_spacing)

    # Create a new blank image with the calculated size
    new_image = Image.new('RGB', (total_width, total_height))

    # Create a new ImageDraw object to draw on the new image
    draw = ImageDraw.Draw(new_image)

    # Loop through the images and draw each one onto the new image
    for i, image in enumerate(image_list):
        # Calculate the x and y position of the current image
        x = (i % images_per_row) * (image_width + horizontal_spacing)
        y = (i // images_per_row) * (image_height + vertical_spacing)
        # Draw the image onto the new image at the calculated position
        new_image.paste(image, (x, y))

    # Save the new image to a file
    cwd = os.getcwd()
    new_image.save(f'{cwd}/grids/{"celeb" if celeb else "cin"}/{model_type}/{steps}.png')

    return new_image, noise_list, prompt_list

def create_horizontal_grid(*images, celeb=False, steps=None, x_labels=None, font_size=20):
    """
    Disclaimer: Written in part by ChatGPT
    Takes as input a list of images and returns them in a horizontal grid.
    """

    # Set the size of the output images
    cwd = os.getcwd()
    image_width, image_height = images[0].size

    # Set the spacing between images
    spacing = 10

    # Calculate the total width of the output image
    total_width = (image_width + spacing) * len(images) - spacing

    # Calculate the total height of the output image
    total_height = image_height + spacing + 40  # Additional space for x-axis names

    # Create a new blank image with the calculated size
    new_image = Image.new('RGBA', (total_width, total_height), (0, 0, 0, 0))

    # Create a new ImageDraw object to draw on the new image
    draw = ImageDraw.Draw(new_image)

    # Loop through the images and draw each one onto the new image
    for i, image in enumerate(images):
        # Calculate the x position of the current image
        x = i * (image_width + spacing)
        # Draw the image onto the new image at the calculated position
        new_image.paste(image, (x, 0))

        # Add x-axis name if provided
        if x_labels is not None and i < len(x_labels):
            x_label = x_labels[i]
            text_width, text_height = draw.textsize(x_label)
            text_x = x + (image_width - text_width) // 2
            text_y = image_height + spacing
            draw.text((text_x, text_y), x_label, fill='black', font=ImageFont.truetype('arial.ttf', font_size))

    # Create the directory if it doesn't exist
    directory = f'{cwd}/grids/hor_{"celeb" if celeb else "cin"}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Check if a file with the same name already exists, and modify the filename accordingly
    if os.path.exists(f'{directory}/{steps}.png'):
        i = 1
        while os.path.exists(f'{directory}/{steps}_{i}.png'):
            i += 1
        filename = f'{steps}_{i}.png'
    else:
        filename = f'{steps}.png'

    # Save the new image with the modified filename
    new_image.save(f'{directory}/{filename}')
    # Return the new image
    return new_image

def create_vertical_grid(*images, celeb=False, steps=None):
    """
    Disclaimer: Written in part by ChatGPT
    Takes as input a list of images and returns them in a vertical grid.
    """
    # Set the size of the output images
    cwd = os.getcwd()
    image_width, image_height = images[0].size

    # Set the spacing between images
    spacing = 10

    # Calculate the total height of the output image
    total_height = (image_height + spacing) * len(images) - spacing

    # Create a new blank image with the calculated size
    new_image = Image.new('RGBA', (image_width, total_height), (0, 0, 0, 0))

    # Create a new ImageDraw object to draw on the new image
    draw = ImageDraw.Draw(new_image)

    # Loop through the images and draw each one onto the new image
    for i, image in enumerate(images):
        # Calculate the y position of the current image
        y = i * (image_height + spacing)
        # Draw the image onto the new image at the calculated position
        new_image.paste(image, (0, y))

    # Create the directory if it doesn't exist
    directory = f'{cwd}/grids/vert_{"celeb" if celeb else "cin"}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Check if a file with the same name already exists, and modify the filename accordingly
    if os.path.exists(f'{directory}/{steps}.png'):
        i = 1
        while os.path.exists(f'{directory}/{steps}_{i}.png'):
            i += 1
        filename = f'{steps}_{i}.png'
    else:
        filename = f'{steps}.png'

    # Save the new image with the modified filename
    new_image.save(f'{directory}/{filename}')
    # Return the new image
    return new_image