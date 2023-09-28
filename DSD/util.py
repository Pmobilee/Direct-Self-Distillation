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

# Receiving base current working directory
cwd = os.getcwd()

def latent_to_img(model, latent):
    """
    Params: model, latent. Task: converts a latent vector to an image
    """
    x_samples_ddim = model.decode_first_stage(latent)
    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
    grid = rearrange(x_samples_ddim, 'b c h w -> (b) c h w')
    grid = make_grid(grid, nrow=1)
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    image = Image.fromarray(grid.astype(np.uint8)) 
    return image

def print_size_of_model(model):
    """
    Params: model. Task: prints the size of the model in MB
    """
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

@torch.no_grad()
def compare_latents(images):
    """
    Compare the latents of a batch of images.
    """
    grid = torch.stack(images, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=2)
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    return Image.fromarray(grid.astype(np.uint8)), grid.astype(np.uint8)
   

@torch.no_grad()
def compare_teacher_student(teacher, sampler_teacher, student, sampler_student, steps=[10], prompt=None, total_steps=64, x0=False):
    """
    Compare the a trained model and an original (teacher). Terms used are teacher and student models, though these may be the same model but at different
    stages of training.
    """
    scale = 3.0
    ddim_eta = 0.0
    images = []
    with torch.no_grad():
        # with teacher.ema_scope():
            for sampling_steps in steps:
                sampler_teacher.make_schedule(ddim_num_steps=sampling_steps, ddim_eta=0.0, verbose=False)
                sampler_student.make_schedule(ddim_num_steps=sampling_steps, ddim_eta=0.0, verbose=False)
                
                if prompt == None:
                    class_image = torch.randint(0, 999, (1,))
                else:
                    class_image = torch.tensor([prompt])

                intermediate_step = None if sampling_steps != 1 else 0
                if x0:
                    uc = None
                    sc=None
                else:
                    sc = student.get_learned_conditioning({student.cond_stage_key: torch.tensor(1*[1000]).to(student.device)})
                    uc = teacher.get_learned_conditioning({teacher.cond_stage_key: torch.tensor(1*[1000]).to(teacher.device)})
                
                #uc = teacher.get_learned_conditioning({teacher.cond_stage_key: torch.tensor(1*[1000]).to(teacher.device)})
                xc = torch.tensor([class_image])
                c = teacher.get_learned_conditioning({teacher.cond_stage_key: xc.to(teacher.device)})
                teacher_samples_ddim, _, x_T_copy, pred_x0_teacher, a_t= sampler_teacher.sample(S=sampling_steps,
                                                    conditioning=c,
                                                    batch_size=1,
                                           
                                                    x_T=None,
                                                    shape=[3, 64, 64],
                                                    verbose=False,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=uc, 
                                                    eta=ddim_eta,
                                                    intermediate_step=intermediate_step ,
                                                    total_steps=sampling_steps,
                                                    steps_per_sampling=sampling_steps)
    
                # x_samples_ddim = teacher.decode_first_stage(_["pred_x0"][-1)
                x_samples_ddim = teacher.decode_first_stage(pred_x0_teacher)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                images.append(x_samples_ddim)
                # with student.ema_scope():
                #sc = student.get_learned_conditioning({student.cond_stage_key: torch.tensor(1*[1000]).to(student.device)})
                c = student.get_learned_conditioning({student.cond_stage_key: xc.to(student.device)})
                student_samples_ddim, _, x_T_delete, pred_x0_student, a_t = sampler_student.sample(S=sampling_steps,
                                                    conditioning=c,
                                                    batch_size=1,
                                                    
                                                    x_T=x_T_copy,
                                                    shape=[3, 64, 64],
                                                    verbose=False,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=sc, 
                                                    eta=ddim_eta,
                                                    intermediate_step=intermediate_step,
                                                    total_steps=sampling_steps,
                                                    steps_per_sampling=sampling_steps)

                x_samples_ddim = student.decode_first_stage(pred_x0_student)
                # x_samples_ddim = teacher.decode_first_stage(_f)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                images.append(x_samples_ddim)

    # from torchmetrics.image.fid import FrechetInceptionDistance
    # print(fid.compute())
    grid = torch.stack(images, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=2)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    return Image.fromarray(grid.astype(np.uint8)), grid.astype(np.uint8)

@torch.no_grad()
def compare_teacher_student_x0(teacher, sampler_teacher, student, sampler_student, steps=[10], prompt=None, total_steps=64,x0=False):
    """
    Compare the a trained model and an original (teacher). Terms used are teacher and student models, though these may be the same model but at different
    stages of training.
    """
    scale = 3.0
    ddim_eta = 0.0
    images = []
    total_steps=max(steps)
    with torch.no_grad():
        # with teacher.ema_scope():
            for sampling_steps in steps:
                sampler_teacher.make_schedule(ddim_num_steps=total_steps, ddim_eta=0.0, verbose=False)
                sampler_student.make_schedule(ddim_num_steps=total_steps, ddim_eta=0.0, verbose=False)
                
                if prompt == None:
                    class_image = torch.randint(0, 999, (1,))
                else:
                    class_image = torch.tensor([prompt])

                intermediate_step = None if sampling_steps != 1 else 0
                if x0:
                    uc = None
                    sc=None
                else:
                    sc = student.get_learned_conditioning({student.cond_stage_key: torch.tensor(1*[1000]).to(student.device)})
                    uc = teacher.get_learned_conditioning({teacher.cond_stage_key: torch.tensor(1*[1000]).to(teacher.device)})
                xc = torch.tensor([class_image])
                c = teacher.get_learned_conditioning({teacher.cond_stage_key: xc.to(teacher.device)})
                teacher_samples_ddim, _, x_T_copy, pred_x0_teacher, a_t= sampler_teacher.sample(S=sampling_steps,
                                                    conditioning=c,
                                                    batch_size=1,
                                           
                                                    x_T=None,
                                                    shape=[3, 64, 64],
                                                    verbose=False,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=uc, 
                                                    eta=ddim_eta,
                                                    intermediate_step=intermediate_step ,
                                                    total_steps=total_steps,
                                                    steps_per_sampling=sampling_steps)

                # x_samples_ddim = teacher.decode_first_stage(_["pred_x0"][-1)
                x_samples_ddim = teacher.decode_first_stage(pred_x0_teacher)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                images.append(x_samples_ddim)
                # with student.ema_scope():
                
                c = student.get_learned_conditioning({student.cond_stage_key: xc.to(student.device)})
                student_samples_ddim, _, x_T_delete, pred_x0_student, a_t = sampler_student.sample(S=sampling_steps,
                                                    conditioning=c,
                                                    batch_size=1,
                                                    
                                                    x_T=x_T_copy,
                                                    shape=[3, 64, 64],
                                                    verbose=False,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=sc, 
                                                    eta=ddim_eta,
                                                    intermediate_step=intermediate_step,
                                                    total_steps=total_steps,
                                                    steps_per_sampling=sampling_steps)

                x_samples_ddim = student.decode_first_stage(pred_x0_student)
                # x_samples_ddim = teacher.decode_first_stage(_f)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                images.append(x_samples_ddim)

    # from torchmetrics.image.fid import FrechetInceptionDistance
    # print(fid.compute())
    grid = torch.stack(images, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=2)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    return Image.fromarray(grid.astype(np.uint8)), grid.astype(np.uint8)

@torch.no_grad()
def compare_teacher_student_retrain(teacher, sampler_teacher, student, sampler_student, steps=[10], prompt=None, total_steps=64):
    """
    Compare the a trained model and an original (teacher). Terms used are teacher and student models, though these may be the same model but at different
    stages of training.
    """
    scale = 3.0
    ddim_eta = 0.0
    images = []
    with torch.no_grad():
        # with teacher.ema_scope():
            for sampling_steps in steps:
                sampler_teacher.make_schedule(ddim_num_steps=sampling_steps, ddim_eta=0.0, verbose=False)
                sampler_student.make_schedule(ddim_num_steps=sampling_steps, ddim_eta=0.0, verbose=False)
                
                if prompt == None:
                    class_image = torch.randint(0, 999, (1,))
                else:
                    class_image = torch.tensor([prompt])

                intermediate_step = None if sampling_steps != 1 else 0
             
                uc = teacher.get_learned_conditioning({teacher.cond_stage_key: torch.tensor(1*[1000]).to(teacher.device)})
                xc = torch.tensor([class_image])
                c = teacher.get_learned_conditioning({teacher.cond_stage_key: xc.to(teacher.device)})
                teacher_samples_ddim, _, x_T_copy, pred_x0_teacher, a_t= sampler_teacher.sample(S=sampling_steps,
                                                    conditioning=c,
                                                    batch_size=1,
                                           
                                                    x_T=None,
                                                    shape=[3, 64, 64],
                                                    verbose=False,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=uc, 
                                                    eta=ddim_eta,
                                                    intermediate_step=intermediate_step ,
                                                    total_steps=sampling_steps,
                                                    steps_per_sampling=sampling_steps)
    
                # x_samples_ddim = teacher.decode_first_stage(_["pred_x0"][-1)
                x_samples_ddim = teacher.decode_first_stage(pred_x0_teacher)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                images.append(x_samples_ddim)
                # with student.ema_scope():
                sc = student.get_learned_conditioning({student.cond_stage_key: torch.tensor(1*[1000]).to(student.device)})
                c = student.get_learned_conditioning({student.cond_stage_key: xc.to(student.device)})
                student_samples_ddim, _, x_T_delete, pred_x0_student, a_t = sampler_student.sample(S=sampling_steps,
                                                    conditioning=c,
                                                    batch_size=1,
                                                    
                                                    x_T=x_T_copy,
                                                    shape=[3, 64, 64],
                                                    verbose=False,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=None, 
                                                    eta=ddim_eta,
                                                    intermediate_step=intermediate_step,
                                                    total_steps=sampling_steps,
                                                    steps_per_sampling=sampling_steps)

                x_samples_ddim = student.decode_first_stage(pred_x0_student)
                # x_samples_ddim = teacher.decode_first_stage(_f)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                images.append(x_samples_ddim)

    # from torchmetrics.image.fid import FrechetInceptionDistance
    # print(fid.compute())
    grid = torch.stack(images, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=2)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    return Image.fromarray(grid.astype(np.uint8)), grid.astype(np.uint8)

@torch.no_grad()
def compare_teacher_student_celeb(teacher, sampler_teacher, student, sampler_student, steps=[10]):
    # print("comapring teacher and student")
    # print("same state dict:", teacher.model.state_dict()['diffusion_model.time_embed.0.weight'][0][0]  == student.model.state_dict()['diffusion_model.time_embed.0.weight'][0][0] )
    scale = 3.0
    ddim_eta = 0.0
    images = []

    total_steps = max(steps)
    with torch.no_grad():
        # with teacher.ema_scope():
            for sampling_steps in steps:
                sampler_teacher.make_schedule(ddim_num_steps=total_steps, ddim_eta=0.0, verbose=False)
                sampler_student.make_schedule(ddim_num_steps=total_steps, ddim_eta=0.0, verbose=False)
                intermediate_step = None if sampling_steps != 1 else 0
                
                
                teacher_samples_ddim, _, x_T_copy, pred_x0, a_t, _= sampler_teacher.sample(S=sampling_steps,

                                                        batch_size=1,
                                                        x_T=None,  
                                                        shape=[3, 64, 64],
                                                        verbose=False,
                                                        unconditional_guidance_scale=scale,
                                           
                                                 
                                                        eta=ddim_eta,
                                                        intermediate_step=intermediate_step,
                                                        total_steps=total_steps,
                                                        steps_per_sampling=sampling_steps)
                
                x_samples_ddim = teacher.decode_first_stage(pred_x0)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                images.append(x_samples_ddim)
               
                # with student.ema_scope():
                student_samples_ddim, _, _, pred_x0_student, _ , _  = sampler_student.sample(S=sampling_steps,
                                           
                                                        batch_size=1,
                                                        x_T= x_T_copy,
                                                        shape=[3, 64, 64],
                                                        verbose=False,
                                                        unconditional_guidance_scale=scale,
                                         
                                                        eta=ddim_eta,
                                                       
                                                        intermediate_step=intermediate_step,
                                                        total_steps=total_steps,
                                                        steps_per_sampling=sampling_steps)

                x_samples_ddim = student.decode_first_stage(pred_x0_student)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                images.append(x_samples_ddim)
    # print("same sample_ddims?", teacher_samples_ddim == student_samples_ddim)
    # print("sample ddim == pred_x0?", teacher_samples_ddim == pred_x0_student)

    grid = torch.stack(images, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=2)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    return Image.fromarray(grid.astype(np.uint8)), grid.astype(np.uint8)


def get_fid(model, sampler, num_imgs, name,instance, steps =[4, 2, 1], x_0=False):
    """
    Calculates the FID score for a given model and sampler. Potentially useful for monitoring training, or comparing distillation
    methods.
    """
    from pytorch_fid import fid_score
    fid_list = []
    if not os.path.exists(f"{cwd}/saved_images/FID/{name}/{instance}"):
        os.makedirs(f"{cwd}/saved_images/FID/{name}/{instance}")
    with torch.no_grad():
        run_name = f"FID/{name}/{instance}/"
        save_images(model, sampler, num_imgs, run_name, steps, verbose=False, x_0=x_0)
        for step in steps:
            fid = fid_score.calculate_fid_given_paths([f"{cwd}/val_saved/real_fid_both.npz", 
            f"{cwd}/saved_images/FID/{name}/{instance}/{step}"], batch_size = 16, device='cuda', dims=2048)
            fid_list.append(fid)
    return fid_list

def get_fid_celeb(model, sampler, num_imgs, name,instance, steps =[4, 2, 1]):
    """
    Calculates the FID score for a given model and sampler. Potentially useful for monitoring training, or comparing distillation
    methods.
    """
    from pytorch_fid import fid_score
    fid_list = []
    if not os.path.exists(f"{cwd}/saved_images/FID/{name}/{instance}"):
        os.makedirs(f"{cwd}/saved_images/FID/{name}/{instance}")
    with torch.no_grad():
        run_name = f"FID/{name}/{instance}/"
        save_images(model, sampler, num_imgs, run_name, steps, verbose=False, celeb=True)
        for step in steps:
            fid = fid_score.calculate_fid_given_paths([f"{cwd}/celeb_64.npz", 
            f"{cwd}/saved_images/FID/{name}/{instance}/{step}"], batch_size = 32, device='cuda', dims=2048)
            fid_list.append(fid)
    return fid_list


def generate_npz(source, destination, batch_size=64, device="cuda", dims=2048):
    paths = [source, destination]
    
    fid_score.save_fid_stats(paths, batch_size=batch_size, device=device, dims=dims)
