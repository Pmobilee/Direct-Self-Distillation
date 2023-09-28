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
import copy
import wandb
import math
import traceback
from pytorch_fid import fid_score
import shutil
import util
import saving_loading
import generate

# Receiving base current working directory
cwd = os.getcwd()

# Scaling allows for reduced precision during gradient calculation, reducing memory consumption but also accuracy
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()

def self_distillation_CIN(student, sampler_student, original, sampler_original, optimizer, scheduler,
            session=None, steps=20, gradient_updates=200, run_name="test",step_scheduler="naive", x0=False):
    """
    Params: student, sampler_student, original, sampler_original, optimizer, scheduler, session=None, steps=20, generations=200, run_name="test", decrease_steps=False, step_scheduler="deterministic"

    Task:Distill a model into itself. This is done by having a (teacher) model distill knowledge into itself. Copies of the original model and sampler 
    are passed in to compare the original untrained version with the distilled model at scheduled intervals.
    """
    NUM_CLASSES = 1000
    ddim_steps_student = steps # Setting the number of steps for the student model
    ddim_eta = 0.0 # Setting the eta value to 0.0 means a deterministic output given the original noise, essential
    # For both the student and the original model, the number of steps is set to the same value. 
    # Technically the original model does not need to be trained, but it is kept for comparison purposes.
    
    ddim_eta = 0.0 # Setting the eta value to 0.0 means a deterministic output given the original noise, essential
    scale = 3.0 # This is $w$ in the paper, the CFG scale. Can be left static or varied as is done occasionally.
    criterion = nn.MSELoss() 

    instance = 0 # Actual instance of student gradient updates
    generation = 0 # The amount of final-step images generated
    averaged_losses = []
    all_losses = []
    
    if step_scheduler == "iterative": # Halve the number of steps from start to 1 with even allocation of gradient updates
        halvings = math.floor(math.log(ddim_steps_student)/math.log(2))
        updates_per_halving = int(gradient_updates / halvings)
        step_sizes = []
        for i in range(halvings):
            step_sizes.append(int((steps) / (2**i)))
        update_list = []
        for i in step_sizes:
            update_list.append(int(updates_per_halving / int(i/ 2))) # /2 because of 2 steps per update
    elif step_scheduler == "naive": # Naive approach, evenly distribute gradient updates over all steps
        step_sizes=[ddim_steps_student]
        update_list=[gradient_updates // int(ddim_steps_student / 2)] # /2 because of 2 steps per update
    elif step_scheduler == "gradual_linear": # Gradually decrease the number of steps to 1, with even allocation of gradient updates
        step_sizes = np.arange(steps, 0, -2)
        update_list = ((1/len(np.append(step_sizes[1:], 1)) * gradient_updates / np.append(step_sizes[1:], 1))).astype(int) * 2 # *2 because of 2 steps per update
    elif step_scheduler == "gradual_exp": # TEMPORARY VERSION, to test if focus on higher steps is better, reverse of the one below
        step_sizes = np.arange(64, 0, -2)
        update_list = np.exp((1 / np.append(step_sizes[1:],1))[::-1]) / np.sum(np.exp((1 / np.append(step_sizes[1:],1))[::-1]))
        update_list = (update_list * gradient_updates /  np.append(step_sizes[1:],1)).astype(int) * 2 # *2 because of 2 steps per update
    # elif step_scheduler == "gradual_exp": # Exponential decrease in number of gradient updates per step
    #     step_sizes = np.arange(64, 0, -2)
    #     update_list = np.exp(1 / np.append(step_sizes[1:],1)) / np.sum(np.exp(1 / np.append(step_sizes[1:],1)))
    #     update_list = ((update_list * 2) * gradient_updates /  np.append(step_sizes[1:],1)).astype(int)
    print(update_list)
    total_steps = max(step_sizes)
    sampler_student.make_schedule(ddim_num_steps=total_steps, ddim_eta=ddim_eta, verbose=False)
    sampler_original.make_schedule(ddim_num_steps=total_steps, ddim_eta=ddim_eta, verbose=False)

    # for param in student.first_stage_model.parameters():
    #     param.requires_grad = False
    # for param in sampler_student.model.first_stage_model.parameters():
    #     param.requires_grad = False
    with torch.no_grad():
        # student.use_ema = False
        # student.train()
        # student.use_ema = True
        # with student.ema_scope(): 
                # if x0:
                #     sc=None
                # else:
                 # Get the learned conditioning
                for i, step in enumerate(step_sizes): # For each step size
                    # if instance != 0 and "gradual" not in step_scheduler:   # Save the model after every step size. Given the large model size, 
                    #                                                         # the gradual versions are not saved each time (steps * 2 * 4.7gb is a lot!)
                    #     util.save_model(sampler_student, optimizer, scheduler, name=step_scheduler, steps=updates, run_name=run_name)
                    updates = int(step / 2) # We take updates as half the step size, because we do 2 steps per update
                    generations = update_list[i] # The number of generations has been determined earlier
                    print("Distilling to:", updates)
                    
                    with tqdm.tqdm(torch.randint(0, NUM_CLASSES, (generations,))) as tepoch: # Take a random class for each generation

                        for i, class_prompt in enumerate(tepoch):
                            generation += 1
                            losses = []       
                            scale = 3.0
                            #scale = np.random.uniform(1.0, 4.0) # Randomly sample a scale for each generation, optional
                            xc = torch.tensor([class_prompt])
                            c_student = sampler_student.model.get_learned_conditioning({student.cond_stage_key: xc.to(sampler_student.model.device)}) # Set to 0 for unconditional, requires pretraining
                            sc = student.get_learned_conditioning({sampler_student.model.cond_stage_key: torch.tensor(1*[1000]).to(sampler_student.model.device)})
                            samples_ddim= None # Setting to None will create a new noise vector for each generation
                            predictions_temp = []
                            
                            for steps in range(updates):
                                # with autocast() and torch.enable_grad(): # For mixed precision training, should not be used for final results
                                    with torch.enable_grad():
                                            instance += 1
                                            
                                            
                                            samples_ddim, pred_x0_student, _, at, v_student= sampler_student.sample_student(S=1,
                                                                                conditioning=c_student,
                                                                                batch_size=1,
                                                                                shape=[3, 64, 64],
                                                                                verbose=False,
                                                                                x_T=samples_ddim, # start noise or teacher output
                                                                                unconditional_guidance_scale=scale,
                                                                                unconditional_conditioning=sc, 
                                                                                eta=ddim_eta,
                                                                                keep_intermediates=False,
                                                                                intermediate_step = steps*2,
                                                                                steps_per_sampling = 1,
                                                                                total_steps = total_steps)
                                            
                                            # Code below first decodes the latent image and then reconstructs it. This is not necessary, but can be used to check if the latent image is correct
                                            # decode_student = student.differentiable_decode_first_stage(pred_x0_student)
                                            # reconstruct_student = torch.clamp((decode_student+1.0)/2.0, min=0.0, max=1.0)
                                            
                                         

                                            with torch.no_grad():
                                                samples_ddim.detach()
                                                samples_ddim, _, _, pred_x0_teacher, _, v = sampler_student.sample(S=1,
                                                                            conditioning=c_student,
                                                                            batch_size=1,
                                                                            shape=[3, 64, 64],
                                                                            verbose=False,
                                                                            x_T=samples_ddim, # output of student
                                                                            unconditional_guidance_scale=scale,
                                                                            unconditional_conditioning=sc, 
                                                                            eta=ddim_eta,
                                                                            keep_intermediates=False,
                                                                            intermediate_step = steps*2+1,
                                                                            steps_per_sampling = 1,
                                                                            total_steps = total_steps)     

                                                # decode_teacher = student.decode_first_stage(pred_x0_teacher)
                                                # reconstruct_teacher = torch.clamp((decode_teacher+1.0)/2.0, min=0.0, max=1.0)
                                        

                                     
                                        
                                            # # AUTOCAST:
                                            # signal = at
                                            # noise = 1 - at
                                            # log_snr = torch.log(signal / noise)
                                            # weight = max(log_snr, 1)
                                            # loss = weight * criterion(pred_x0_student, pred_x0_teacher.detach())
                                            # scaler.scale(loss).backward()
                                            # scaler.step(optimizer)
                                            # scaler.update()
                                            # torch.nn.utils.clip_grad_norm_(sampler_student.model.parameters(), 1)
                                            # losses.append(loss.item())

                                            optimizer.zero_grad()
                                            # # NO AUTOCAST:
                                            signal = at
                                            noise = 1 - at
                                            log_snr = torch.log(signal / noise)
                                            weight = max(log_snr, 1)
                                            loss = weight * criterion(pred_x0_student, pred_x0_teacher.detach())     
                                            # loss = weight * criterion(v_student, v.detach())                    
                                            loss.backward()
                                            torch.nn.utils.clip_grad_norm_(sampler_student.model.parameters(), 1)
                                            optimizer.step()
                                            scheduler.step()
                                            
                                            losses.append(loss.item())

                                            # if session != None and instance % 10000 == 0 and generation > 0:
                                            #     fids = util.get_fid(student, sampler_student, num_imgs=100, name=run_name, instance = instance+1, steps=[64, 32, 16, 8, 4, 2, 1])
                                            #     session.log({"fid_64":fids[0]})
                                            #     session.log({"fid_32":fids[1]})
                                            #     session.log({"fid_16":fids[2]})
                                            #     session.log({"fid_8":fids[3]})
                                            #     session.log({"fid_4":fids[4]})
                                            #     session.log({"fid_2":fids[5]})
                                            #     session.log({"fid_1":fids[6]})
                                            
                            if session != None and generation > 0 and generation % 5 == 0: # or instance==1:

                                with torch.no_grad():
                                        # the x0 version keeps max denoising steps to 64
                                        images, _ = util.compare_teacher_student_x0(original, sampler_original, student, sampler_student, steps=[16, 8,  4, 1], prompt=992, x0=x0)
                                        images = wandb.Image(_, caption="left: Teacher, right: Student")
                                        wandb.log({"pred_x0": images})

                                        # Optional; compare the images but also change the denoising schedule
                                        images, _ = util.compare_teacher_student(original, sampler_original, student, sampler_student, steps=[16, 8,  4, 1], prompt=992,x0=x0)
                                        images = wandb.Image(_, caption="left: Teacher, right: Student")
                                        wandb.log({"with_sched": images})

                                        # Important: Reset the schedule, as compare_teacher_student changes max steps. 
                                        sampler_student.make_schedule(ddim_num_steps=total_steps, ddim_eta=ddim_eta, verbose=False)
                                        sampler_original.make_schedule(ddim_num_steps=total_steps, ddim_eta=ddim_eta, verbose=False)

                            # # Use if you want to base the schedule on FID
                            # if generation > 0 and generation % 20 == 0 and ddim_steps_student != 1 and step_scheduler=="FID":
                            #     fid = util.get_fid(student, sampler_student, num_imgs=100, name=run_name, 
                            #                 instance = instance, steps=[ddim_steps_student])
                            #     if fid[0] <= current_fid[0] * 0.9 and decrease_steps==True:
                            #         print(fid[0], current_fid[0])
                            #         if ddim_steps_student in [16, 8, 4, 2, 1]:
                            #             name = "intermediate"
                            #             saving_loading.save_model(sampler_student, optimizer, scheduler, name, steps * 2, run_name)
                            #         if ddim_steps_student != 2:
                            #             ddim_steps_student -= 2
                            #             updates -= 1
                            #         else:
                            #             ddim_steps_student = 1
                            #             updates = 1    
                            #         current_fid = fid
                            #         print("steps decreased:", ddim_steps_student)    
                            
                            all_losses.extend(losses)
                            averaged_losses.append(sum(losses) / len(losses))
                            if session != None:
                                session.log({"generation_loss":averaged_losses[-1]})
                            tepoch.set_postfix(epoch_loss=averaged_losses[-1])

                if step_scheduler == "naive" or "gradual" in step_scheduler: # Save the final model, since we skipped all the intermediate steps
                    util.save_model(sampler_student, optimizer, scheduler, name=step_scheduler, steps=updates, run_name=run_name)

                                                                           
def self_distillation_CELEB(student, sampler_student, original, sampler_original, optimizer, scheduler,
        session=None, steps=20, generations=200, run_name="test",step_scheduler="deterministic"):
    """
    Distill a model into itself. This is done by having a (teacher) model distill knowledge into itself. Copies of the original model and sampler 
    are passed in to compare the original untrained version with the distilled model at scheduled intervals.
    """
    NUM_CLASSES = 1000
    ddim_steps_student = steps # Setting the number of steps for the student model
    ddim_eta = 0.0
    # For both the student and the original model, the number of steps is set to the same value. 
    # Technically the original model does not need to be trained, but it is kept for comparison purposes.
    sampler_student.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)
    sampler_original.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)
     # Setting the eta value to 0.0 means a deterministic output given the original noise, essential
    scale = 3.0 # This is $w$ in the paper, the CFG scale. Can be left static or varied as is done occasionally.
    criterion = nn.MSELoss() 

    instance = 0 # Actual instance of student gradient updates
    generation = 0 # The amount of final-step images generated
    averaged_losses = []
    all_losses = []
    gradient_updates = generations
    if step_scheduler == "iterative": # Halve the number of steps from start to 1 with even allocation of gradient updates
        halvings = math.floor(math.log(ddim_steps_student)/math.log(2))
        updates_per_halving = int(gradient_updates / halvings)
        step_sizes = []
        for i in range(halvings):
            step_sizes.append(int((steps) / (2**i)))
        update_list = []
        for i in step_sizes:
            update_list.append(int(updates_per_halving / int(i/ 2))) # /2 because of 2 steps per update
    elif step_scheduler == "naive": # Naive approach, evenly distribute gradient updates over all steps
        step_sizes=[ddim_steps_student]
        update_list=[gradient_updates // int(ddim_steps_student / 2)] # /2 because of 2 steps per update
    elif step_scheduler == "gradual_linear": # Gradually decrease the number of steps to 1, with even allocation of gradient updates
        step_sizes = np.arange(steps, 0, -2)
        update_list = ((1/len(np.append(step_sizes[1:], 1)) * gradient_updates / np.append(step_sizes[1:], 1))).astype(int) * 2 # *2 because of 2 steps per update
    elif step_scheduler == "gradual_exp": # TEMPORARY VERSION, to test if focus on higher steps is better, reverse of the one below
        step_sizes = np.arange(64, 0, -2)
        update_list = np.exp((1 / np.append(step_sizes[1:],1))[::-1]) / np.sum(np.exp((1 / np.append(step_sizes[1:],1))[::-1]))
        update_list = (update_list * gradient_updates /  np.append(step_sizes[1:],1)).astype(int) * 2 # *2 because of 2 steps per update
    # elif step_scheduler == "gradual_exp": # Exponential decrease in number of gradient updates per step
    #     step_sizes = np.arange(64, 0, -2)
    #     update_list = np.exp(1 / np.append(step_sizes[1:],1)) / np.sum(np.exp(1 / np.append(step_sizes[1:],1)))
    #     update_list = ((update_list * 2) * gradient_updates /  np.append(step_sizes[1:],1)).astype(int)

    # Weird structure for gradient calculations in an  attempt to save memory, Celeb wants a lot for some reason
    with torch.no_grad():
        with student.ema_scope():              
      
            for i, step in enumerate(step_sizes):
                if instance != 0 and "gradual" not in step_scheduler:   # Save the model after every step size. Given the large model size, 
                                                                        # the gradual versions are not saved each time (steps * 2 * 4.7gb is a lot!)
                    util.save_model(sampler_student, optimizer, scheduler, name=step_scheduler, steps=updates, run_name=run_name)
                updates = int(step / 2) # We take updates as half the step size, because we do 2 steps per update
                generations = update_list[i] # The number of generations has been determined earlier
                print("Distilling to:", updates)

                with tqdm.tqdm(range(generations)) as tepoch:
                    for j in tepoch:
                        generation += 1
                        losses = []        
            
                        samples_ddim = None # Setting to None will create a new noise vector for each generation
                        predictions_temp = []
                        for steps in range(updates):  
                            with autocast and torch.enable_grad(): # For mixed precision training, should not be used for final results
                                instance += 1
                                optimizer.zero_grad()

                                samples_ddim, pred_x0_student, _, at, _= sampler_student.sample_student(S=1,
                                                                    conditioning=None,
                                                                    batch_size=1,
                                                                    shape=[3, 64, 64],
                                                                    verbose=False,
                                                                    x_T=samples_ddim, # start noise or teacher output
                                                                    unconditional_guidance_scale=scale,
                                                                    unconditional_conditioning=None, 
                                                                    eta=ddim_eta,
                                                                    keep_intermediates=False,
                                                                    intermediate_step = steps*2,
                                                                    steps_per_sampling = 1,
                                                                    total_steps = step)
                            

                            with torch.no_grad():
                                
                                samples_ddim, _, _, pred_x0_teacher, _ , _ = sampler_student.sample(S=1,
                                                                conditioning=None,
                                                                batch_size=1,
                                                                shape=[3, 64, 64],
                                                                verbose=False,
                                                                x_T=samples_ddim, #output of student
                                                                unconditional_guidance_scale=scale,
                                                                unconditional_conditioning=None, 
                                                                eta=ddim_eta,
                                                                keep_intermediates=False,
                                                                intermediate_step = steps*2+1,
                                                                steps_per_sampling = 1,
                                                                total_steps = step)     
                            
                            
                        
                            with torch.enable_grad():    
                            
                                # # AUTOCAST:
                                # signal = at
                                # noise = 1 - at
                                # log_snr = torch.log(signal / noise)
                                # weight = max(log_snr, 1)
                                # loss = weight * criterion(pred_x0_student, pred_x0_teacher.detach())
                                # scaler.scale(loss).backward()
                                # scaler.step(optimizer)
                                # scaler.update()
                                # torch.nn.utils.clip_grad_norm_(sampler_student.model.parameters(), 1)
                                # losses.append(loss.item())

                                # No AutoCast:
                                signal = at
                                noise = 1 - at
                                log_snr = torch.log(signal / noise)
                                weight = max(log_snr, 1)
                                loss = weight * criterion(pred_x0_student, pred_x0_teacher.detach())
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(sampler_student.model.parameters(), 1)
                                optimizer.step()
                                scheduler.step()
                                
                                losses.append(loss.item())

                            # if session != None and instance % 10000 == 0 and generation > 0:
                            #     fids = util.get_fid_celeb(student, sampler_student, num_imgs=100, name=run_name, instance = instance+1, steps=[64, 32, 16, 8, 4, 2, 1])
                            #     session.log({"fid_64":fids[0]})
                            #     session.log({"fid_32":fids[1]})
                            #     session.log({"fid_16":fids[2]})
                            #     session.log({"fid_8":fids[3]})
                            #     session.log({"fid_4":fids[4]})
                            #     session.log({"fid_2":fids[5]})
                            #     session.log({"fid_1":fids[6]})

                        # if generation > 0 and generation % 20 == 0 and ddim_steps_student != 1 and step_scheduler=="FID":
                        #     fid = util.get_fid(student, sampler_student, num_imgs=100, name=run_name, 
                        #                 instance = instance, steps=[ddim_steps_student])
                        #     if fid[0] <= current_fid[0] * 0.9 and decrease_steps==True:
                        #         print(fid[0], current_fid[0])
                        #         if ddim_steps_student in [16, 8, 4, 2, 1]:
                        #             name = "intermediate"
                        #             saving_loading.save_model(sampler_student, optimizer, scheduler, name, steps * 2, run_name)
                        #         if ddim_steps_student != 2:
                        #             ddim_steps_student -= 2
                        #             updates -= 1
                        #         else:
                        #             ddim_steps_student = 1
                        #             updates = 1    
                        #         current_fid = fid
                        #         print("steps decresed:", ddim_steps_student)    

                        if session != None:
                            with torch.no_grad():
                                if session != None and generation % 10 == 0:
                                    images, _ = util.compare_teacher_student_celeb(original, sampler_original, student, sampler_student, steps=[64, 32, 16, 8,  4, 2, 1])
                                    images = wandb.Image(_, caption="left: Teacher, right: Student")
                                    wandb.log({"pred_x0": images})
                                    torch.cuda.empty_cache()
                        
                        all_losses.extend(losses)
                        averaged_losses.append(sum(losses) / len(losses))
                        if session != None:
                            session.log({"generation_loss":averaged_losses[-1]})
                        tepoch.set_postfix(epoch_loss=averaged_losses[-1])

            if step_scheduler == "naive" or "gradual" in step_scheduler: # Save the final model, since we skipped all the intermediate steps
                    util.save_model(sampler_student, optimizer, scheduler, name=step_scheduler, steps=updates, run_name=run_name)