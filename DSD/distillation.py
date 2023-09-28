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

print("Not using autocast by default, if desired, uncomment this in distillation.py")
# from torch.cuda.amp import GradScaler, autocast
# scaler = GradScaler()

@torch.enable_grad()
def train_student_from_dataset(model, sampler, dataset, student_steps, optimizer, scheduler, early_stop=False, session=None, run_name="test"):
    """
    Deprecated: Train a student model from a pre-generated dataset. Not updated to current methods, possibly useful as a template
    """
    device = torch.device("cuda")
    model.requires_grad=True
    sampler.requires_grad=True
    for param in sampler.model.parameters():
        param.requires_grad = True

    for param in model.model.parameters():
        param.requires_grad = True
    MSEloss = model.criterion
    ddim_steps_student = student_steps
    STUDENT_STEPS = 1
    ddim_eta = 0.0
    scale = 3.0

    averaged_losses = []
    teacher_samples = list()
    criterion = nn.MSELoss()
    optimizer = optimizer
    generation = 0
    instance = 0
    with torch.no_grad():
        
        with model.ema_scope():
            uc = model.get_learned_conditioning({model.cond_stage_key: torch.tensor(1*[1000]).to(model.device)})

            with tqdm.tqdm(range(len(dataset))) as tepoch:
                    for i, _ in enumerate(tepoch):
                        class_prompt = dataset[str(i)]["class"]
                        losses = []
                        sampler.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)
                        xc = torch.tensor([class_prompt])
                        c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                        sampler.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)
                        c_student = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                        generation += 1
                        for steps, x_T in enumerate(dataset[str(i)]["intermediates"]):
                            instance += 0
                            if steps == ddim_steps_student:
                                continue
                            with torch.enable_grad():
                                optimizer.zero_grad()
                                x_T.requires_grad=True
                                
                                samples_ddim_student, student_intermediate, x_T_copy, a_t = sampler.sample_student(S=STUDENT_STEPS,
                                                                conditioning=c_student,
                                                                batch_size=1,
                                                                shape=[3, 64, 64],
                                                                verbose=False,
                                                                x_T=x_T,
                                                             
                                                                unconditional_guidance_scale=scale,
                                                                unconditional_conditioning=uc, 
                                                                eta=ddim_eta,
                                                                keep_intermediates=False,
                                                                intermediate_step = steps*STUDENT_STEPS,
                                                                steps_per_sampling = STUDENT_STEPS,
                                                                total_steps = ddim_steps_student)
                                
                                x_T_student = student_intermediate["x_inter"][-1]
                                # loss = criterion(x_T_student, dataset[str(i)]["intermediates"][steps+1])
                                loss = max(math.log(a_t / (1-a_t)), 1) *  criterion(x_T_student, dataset[str(i)]["intermediates"][steps+1])
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(sampler_student.model.parameters(), 1)
                                optimizer.step()
                                scheduler.step()
                                # x_T.detach()
                                losses.append(loss.item())
                                if session != None:
                                    session.log({"loss":loss.item()})  
                                if instance % 10000 == 0 and generation > 2:
                                    saving_loading.save_model(sampler, optimizer, scheduler, name=f"intermediate_{instance}", steps=student_steps, run_name=run_name)
                            
                                

                        # print("Loss: ", round(sum(losses) / len(losses), 5), end= " - ")
                        averaged_losses.append(sum(losses) / len(losses))
                        tepoch.set_postfix(loss=averaged_losses[-1])
                        if session != None:
                            session.log({"generation_loss":averaged_losses[-1]})
                            session.log({"generation":generation})

    

@torch.enable_grad()
def train_student_from_dataset_celeb(model, sampler, dataset, student_steps, optimizer, scheduler, early_stop=False, session=None, run_name="test"):
    """
    Deprecated: Train a student model from a pre-generated dataset. Not updated to current methods, possibly useful as a template
    """

    device = torch.device("cuda")
    model.requires_grad=True
    sampler.requires_grad=True
    for param in sampler.model.parameters():
        param.requires_grad = True

    for param in model.model.parameters():
        param.requires_grad = True
    MSEloss = model.criterion
    ddim_steps_student = student_steps
    STUDENT_STEPS = 1
    ddim_eta = 0.0
    scale = 3.0

    averaged_losses = []
    teacher_samples = list()
    criterion = nn.MSELoss()
    optimizer = optimizer
    generation = 0
    instance = 0
    with torch.no_grad():
        
        with model.ema_scope():
            uc = model.get_learned_conditioning({model.cond_stage_key: torch.tensor(1*[1000]).to(model.device)})

            with tqdm.tqdm(range(len(dataset))) as tepoch:
                    for i, _ in enumerate(tepoch):
                        
                        losses = []
                        sampler.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)
                        sampler.make_schedule(ddim_num_steps=ddim_steps_student, ddim_eta=ddim_eta, verbose=False)
                        generation += 1
                        for steps, x_T in enumerate(dataset[str(i)]["intermediates"]):
                            instance += 0
                            if steps == ddim_steps_student:
                                continue
                            with torch.enable_grad():
                                optimizer.zero_grad()
                                x_T.requires_grad=True
                                
                                samples_ddim_student, student_intermediate, x_T_copy, a_t = sampler.sample_student(S=STUDENT_STEPS,   
                                                                batch_size=1,
                                                                shape=[3, 64, 64],
                                                                verbose=False,
                                                                x_T=x_T,
                                                            
                                                                unconditional_guidance_scale=scale,
                                                                unconditional_conditioning=uc, 
                                                                eta=ddim_eta,
                                                                keep_intermediates=False,
                                                                intermediate_step = steps*STUDENT_STEPS,
                                                                steps_per_sampling = STUDENT_STEPS,
                                                                total_steps = ddim_steps_student)
                                
                                x_T_student = student_intermediate["x_inter"][-1]
                                # loss = criterion(x_T_student, dataset[str(i)]["intermediates"][steps+1])
                                loss = max(math.log(a_t / (1-a_t)), 1) *  criterion(x_T_student, dataset[str(i)]["intermediates"][steps+1])
                                
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(sampler_student.model.parameters(), 1)
                                optimizer.step()
                                scheduler.step()
                                # x_T.detach()
                                losses.append(loss.item())
                                if session != None:
                                    session.log({"loss":loss.item()})  
                                if instance % 10000 == 0 and generation > 2:
                                    saving_loading.save_model(sampler, optimizer, scheduler, name=f"intermediate_{instance}", steps=student_steps, run_name=run_name)
                            
                                

                        # print("Loss: ", round(sum(losses) / len(losses), 5), end= " - ")
                        averaged_losses.append(sum(losses) / len(losses))
                        tepoch.set_postfix(loss=averaged_losses[-1])
                        if session != None:
                            session.log({"generation_loss":averaged_losses[-1]})
                            session.log({"generation":generation})


def teacher_train_student(teacher, sampler_teacher, student, sampler_student, optimizer, scheduler, session=None, steps=20, generations=200, early_stop=True, run_name="test", cas=False, x0=False):
    """
    Params: teacher, sampler_teacher, student, sampler_student, optimizer, scheduler, session=None, steps=20, generations=200, early_stop=True, run_name="test". 
    Task: trains the student model using the identical teacher model as a guide. Not used in direct self-distillation where a teacher distills into itself.
    """
    NUM_CLASSES = 1000
    generations = generations
    ddim_steps_teacher = steps
    ddim_steps_teacher = int(ddim_steps_teacher / 2)
    TEACHER_STEPS = 2
    STUDENT_STEPS = 1
    ddim_eta = 0.0
    scale = 3.0
    updates = int(ddim_steps_teacher / TEACHER_STEPS)
    optimizer=optimizer
    averaged_losses = []
    criterion = nn.MSELoss()
    instance = 0
    generation = 0
    all_losses = []

    with torch.no_grad():
        with teacher.ema_scope():
                sampler_teacher.make_schedule(ddim_num_steps=ddim_steps_teacher, ddim_eta=ddim_eta, verbose=False)
                sampler_student.make_schedule(ddim_num_steps=ddim_steps_teacher, ddim_eta=ddim_eta, verbose=False)

                if x0:
                    sc=None
                    uc=None
                else:
                    uc = teacher.get_learned_conditioning(
                            {teacher.cond_stage_key: torch.tensor(1*[1000]).to(teacher.device)}
                            )
                    sc = student.get_learned_conditioning({student.cond_stage_key: torch.tensor(1*[1000]).to(student.device)})
                with tqdm.tqdm(torch.randint(0, NUM_CLASSES, (generations,))) as tepoch:
                    for i, class_prompt in enumerate(tepoch):

                        generation += 1
                        losses = []        
                        xc = torch.tensor([class_prompt])
                        c = teacher.get_learned_conditioning({teacher.cond_stage_key: xc.to(teacher.device)})
                        c_student = student.get_learned_conditioning({student.cond_stage_key: xc.to(student.device)})      
                        samples_ddim_teacher = None
                        predictions_temp = []

                        for steps in range(updates):        
                                    instance += 1
                                    samples_ddim_teacher, teacher_intermediate, x_T, pred_x0_teacher, a_t_teacher, _ = sampler_teacher.sample(S=TEACHER_STEPS,
                                                                    conditioning=c,
                                                                    batch_size=1,
                                                                    shape=[3, 64, 64],
                                                                    verbose=False,
                                                                    x_T=samples_ddim_teacher,
                                                                
                                                                    # quantize_x0 = True,
                                                                    unconditional_guidance_scale=scale,
                                                                    unconditional_conditioning=uc, 
                                                                    eta=ddim_eta,
                                                                    keep_intermediates=False,
                                                                    intermediate_step = steps*2,
                                                                    steps_per_sampling = TEACHER_STEPS,
                                                                    total_steps = ddim_steps_teacher)      
                                    
                                    with torch.enable_grad():
                                        with student.ema_scope():
                                            optimizer.zero_grad()
                                            samples, pred_x0_student, st, at, _ = sampler_student.sample_student(S=STUDENT_STEPS,
                                                                            conditioning=c_student,
                                                                            batch_size=1,
                                                                            shape=[3, 64, 64],
                                                                            verbose=False,
                                                                            x_T=x_T,
                                                                         
                                                                            # quantize_x0 = True,
                                                                            unconditional_guidance_scale=scale,
                                                                            unconditional_conditioning=sc, 
                                                                            eta=ddim_eta,
                                                                            keep_intermediates=False,
                                                                            intermediate_step = steps*2,
                                                                            steps_per_sampling = STUDENT_STEPS,
                                                                            total_steps = ddim_steps_teacher)
                                            
                                            # with autocast():    
                                            #     # AUTOCAST:
                                            #     signal = at
                                            #     noise = 1 - at
                                            #     log_snr = torch.log(signal / noise)
                                            #     weight = max(log_snr, 1)
                                            #     loss = weight * criterion(pred_x0_student, pred_x0_teacher.detach())
                                            #     scaler.scale(loss).backward()
                                            #     scaler.step(optimizer)
                                            #     scaler.update()
                                            #     # torch.nn.utils.clip_grad_norm_(sampler_student.model.parameters(), 1)
                                                
                                            #     scheduler.step()
                                            #     losses.append(loss.item())

                                            # NO AUTOCAST:
                                            signal = at
                                            noise = 1 - at
                                            log_snr = torch.log(signal / noise)
                                            weight = max(log_snr, 1)
                                            loss = weight * criterion(pred_x0_student, pred_x0_teacher)
                                            loss.backward()
                                            torch.nn.utils.clip_grad_norm_(sampler_student.model.parameters(), 1)
                                            optimizer.step()
                                            scheduler.step()    
                                            losses.append(loss.item())
                                                
                                    if session != None and generation % 200 == 0 and generation > 0:
                                        x_T_teacher_decode = sampler_teacher.model.decode_first_stage(pred_x0_teacher)
                                        teacher_target = torch.clamp((x_T_teacher_decode+1.0)/2.0, min=0.0, max=1.0)
                                        x_T_student_decode = sampler_student.model.decode_first_stage(pred_x0_student.detach())
                                        student_target  = torch.clamp((x_T_student_decode +1.0)/2.0, min=0.0, max=1.0)
                                        predictions_temp.append(teacher_target)
                                        predictions_temp.append(student_target)
                            
                        if session != None and generation > 0 and generation % 25 == 0:
                            with torch.no_grad():
                                images, _ = util.compare_teacher_student(teacher, sampler_teacher, student, sampler_student, steps=[64, 32, 16, 8,  4, 2, 1], prompt=992, x0=x0)
                                images = wandb.Image(_, caption="left: Teacher, right: Student")
                                wandb.log({"pred_x0": images}) 
                                sampler_student.make_schedule(ddim_num_steps=ddim_steps_teacher, ddim_eta=ddim_eta, verbose=False)
                                sampler_teacher.make_schedule(ddim_num_steps=ddim_steps_teacher, ddim_eta=ddim_eta, verbose=False)

                        if session != None:
                            with torch.no_grad():
                                if generation > 0 and generation % 200 == 0 and session !=None:
                                    img, grid = util.compare_latents(predictions_temp)
                                    images = wandb.Image(grid, caption="left: Teacher, right: Student")
                                    wandb.log({"Inter_Comp": images})
                                    del img, grid, predictions_temp, x_T_student_decode, x_T_teacher_decode, student_target, teacher_target
                                    torch.cuda.empty_cache()

                        all_losses.extend(losses)
                        # print(scheduler.get_last_lr())
                        averaged_losses.append(sum(losses) / len(losses))
                        if session != None:
                            session.log({"generation_loss":averaged_losses[-1]})
                        tepoch.set_postfix(epoch_loss=averaged_losses[-1])
                        



def teacher_train_student_celeb(teacher, sampler_teacher, student, sampler_student, optimizer, scheduler, session=None, steps=20, generations=200, early_stop=True, run_name="test"):
    """
    Params: teacher, sampler_teacher, student, sampler_student, optimizer, scheduler, session=None, steps=20, generations=200, early_stop=True, run_name="test". 
    Task: trains the student model using the identical teacher model as a guide. Not used in direct self-distillation where a teacher distills into itself.
    """
    generations = generations
    intermediate_generation_save = generations // 2
    intermediate_generation_compare = generations // 4

    ddim_steps_teacher = steps
    ddim_steps_teacher = int(ddim_steps_teacher / 2)
    TEACHER_STEPS = 2
    STUDENT_STEPS = 1
    ddim_eta = 0.0
    scale = 3.0
    updates = int(ddim_steps_teacher / TEACHER_STEPS)
    optimizer=optimizer
    averaged_losses = []
    teacher_samples = list()
    criterion = nn.MSELoss()
    instance = 0
    generation = 0
    all_losses = []
    sampler_teacher.make_schedule(ddim_num_steps=ddim_steps_teacher, ddim_eta=ddim_eta, verbose=False)
    sampler_student.make_schedule(ddim_num_steps=ddim_steps_teacher, ddim_eta=ddim_eta, verbose=False)

    with torch.no_grad(): # and autocast():
        with teacher.ema_scope():
                with tqdm.tqdm(torch.randint(0, 1000, (generations,))) as tepoch:
                    for i, _ in enumerate(tepoch):    
                        generation += 1
                        losses = []        
                        samples_ddim_teacher = None
                        predictions_temp = []
                        for steps in range(updates):          
                            instance += 1
                            samples_ddim_teacher, _, x_T, pred_x0_teacher, _ , _ = sampler_teacher.sample(S=TEACHER_STEPS,
                                                            conditioning=None,
                                                            batch_size=1,
                                                            shape=[3, 64, 64],
                                                            verbose=False,
                                                            x_T=samples_ddim_teacher,
                                                            unconditional_guidance_scale=scale,
                                                            unconditional_conditioning=None, 
                                                            eta=ddim_eta,
                                                            
                                                            keep_intermediates=False,
                                                            quantize_x0=False,
                                                            intermediate_step = steps*2,
                                                            steps_per_sampling = TEACHER_STEPS,
                                                            total_steps = ddim_steps_teacher)     

                            with torch.enable_grad(): # and autocast():
                                with student.ema_scope():
                                    optimizer.zero_grad()
                                    samples_ddim, pred_x0_student, _, at, _ = sampler_student.sample_student(S=STUDENT_STEPS,
                                                                    conditioning=None,
                                                                    batch_size=1,
                                                                    shape=[3, 64, 64],
                                                                    verbose=False,
                                                                    x_T=x_T,
                                                                    unconditional_guidance_scale=scale,
                                                                    unconditional_conditioning=None, 
                                                                    quantize_x0=False,
                                                            
                                                                    eta=ddim_eta,
                                                                    keep_intermediates=False,
                                                                    intermediate_step = steps*2,
                                                                    steps_per_sampling = STUDENT_STEPS,
                                                                    total_steps = ddim_steps_teacher)
                                    
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
                                    
                            if session != None:
                                if generation > 0 and generation % 200 == 0:
                                    x_T_teacher_decode = sampler_teacher.model.decode_first_stage(pred_x0_teacher.detach())
                                    teacher_target = torch.clamp((x_T_teacher_decode+1.0)/2.0, min=0.0, max=1.0)
                                    x_T_student_decode = sampler_student.model.decode_first_stage(pred_x0_student.detach())
                                    student_target  = torch.clamp((x_T_student_decode +1.0)/2.0, min=0.0, max=1.0)
                                    predictions_temp.append(teacher_target)
                                    predictions_temp.append(student_target)

                            if session != None and instance % 100 == 0:
                                with torch.no_grad():
                                    images, _ = util.compare_teacher_student_celeb(teacher, sampler_teacher, student, sampler_student, steps=[1, 2, 4, 8, 16, 32, 64])
                                    images = wandb.Image(_, caption="left: Teacher, right: Student")
                                    wandb.log({"pred_x0": images})
                                    sampler_student.make_schedule(ddim_num_steps=ddim_steps_teacher, ddim_eta=ddim_eta, verbose=False)
                                    sampler_teacher.make_schedule(ddim_num_steps=ddim_steps_teacher, ddim_eta=ddim_eta, verbose=False)  
                        
                        if session != None:
                            with torch.no_grad():
                                if generation > 0 and generation % 200 == 0 and session !=None:
                                    img, grid = util.compare_latents(predictions_temp)
                                    images = wandb.Image(grid, caption="left: Teacher, right: Student")
                                    wandb.log({"Inter_Comp": images})
                                    del img, grid, predictions_temp, x_T_student_decode, x_T_teacher_decode, student_target, teacher_target
                                    torch.cuda.empty_cache()
                                
                        all_losses.extend(losses)
                        averaged_losses.append(sum(losses) / len(losses))
                        if session != None:
                            session.log({"generation_loss":averaged_losses[-1]})
                        tepoch.set_postfix(epoch_loss=averaged_losses[-1])

    

def distill(args, config, original_model_path, start_trained=False):
    """
    Distill a model into a smaller model (TSD). This is done by training a student model to match the teacher model with identical initialization.
    This is not direct self-distillation as the teacher model does not distill into itself, but rather into a student model.
    """

    ddim_steps=args.steps
    generations=args.updates 
    run_name=args.name 
    lr=args.learning_rate 
    cas=args.cas 
    compare=args.compare
    use_wandb=args.wandb
    halvings = math.floor(math.log(ddim_steps)/math.log(2)) + 1
    updates_per_half = int(generations / halvings)
    ddim_step_list = []
    for i in range(halvings):
        ddim_step_list.append(2**i)
    ddim_step_list.reverse()
    print(f"Performing TSD for steps: {ddim_step_list}")

    for index, step in enumerate(ddim_step_list):
        steps = int(step / 2)
        model_generations = updates_per_half // steps
        config_path=config
        if index == 0 and start_trained != True:
            model_path=original_model_path
            teacher, sampler_teacher, student, sampler_student = saving_loading.create_models(config_path, model_path, student=True)
            optimizer, scheduler = saving_loading.get_optimizer(sampler_student, iterations=generations, warmup_epochs=100, lr=lr)
            print("Loading New Student and teacher:", step)
        else:
            model_path = f"{cwd}/data/trained_models/TSD/{run_name}/{step}.pt"
            print("Loading New Student and teacher:", step)
            teacher, sampler_teacher, optimizer, scheduler = saving_loading.load_trained(model_path, config_path)
            student = copy.deepcopy(teacher)
            sampler_student = DDIMSampler(student)
        
        if index == 0 and use_wandb:
            wandb_session = util.wandb_log(name=run_name, lr=lr, model=student, tags=["TSD"], 
            notes=f"Teacher-Student Distillation from {steps} steps with {generations} weight updates",  project="Self-Distillation")
            wandb_session.log_code(".")
    
        teacher_train_student(teacher, sampler_teacher, student, sampler_student, optimizer, scheduler, steps=step, generations=model_generations, 
                              early_stop=False, session=wandb_session, run_name=run_name, cas=cas, x0=args.predict)
        
        saving_loading.save_model(sampler_student, optimizer, scheduler, name="TSD", steps=steps, run_name=run_name)
        if compare and use_wandb:
            images, grid = util.compare_teacher_student(teacher, sampler_teacher, student, sampler_student, steps=[1, 2, 4, 8, 16, 32, 64], x0=args.predict)
            images = wandb.Image(grid, caption="left: Teacher, right: Student")
            wandb.log({"Comparison": images})
        
        del teacher, sampler_teacher, student, sampler_student, optimizer, scheduler
        torch.cuda.empty_cache()
    if use_wandb:
        wandb.finish()

def retrain(ddim_steps, generations, run_name, config, original_model_path, lr, start_trained=False, cas=False, compare=True, use_wandb=True):
    """
    Distill a model into a smaller model. This is done by training a student model to match the teacher model with identical initialization.
    This is not direct self-distillation as the teacher model does not distill into itself, but rather into a student model.
    """

    print(f"Performing retrain")
    config_path=config
    model_path=original_model_path
    teacher, sampler_teacher, = saving_loading.create_models(config_path, model_path, student=False)
    print("Loading teacher and student.")

    try:
        model_path=f"{cwd}/retrained.pt"
    except:
        print("Please specify the correct retrained model path. Current path:", model_path)
        exit()

    student, sampler_student, = saving_loading.create_models(config_path, model_path, student=False)
    wandb_session = util.wandb_log(name=run_name, lr=lr, model=student, tags=["retrain"], 
    notes=f"Retrain",  project="Self-Distillation")
    wandb_session.log_code(".")

    warmup_epochs = 1000  # The number of initial iterations to linearly increase the learning rate
    optimizer, scheduler = util.get_optimizer(sampler_teacher, iterations=generations, warmup_epochs=warmup_epochs, eta_min=lr, lr=lr)
    teacher_retrain_student(teacher, sampler_teacher, student, sampler_student, optimizer, scheduler, steps=ddim_steps, generations=generations, 
                            early_stop=False, session=wandb_session, run_name=run_name, cas=cas)
    
    saving_loading.save_model(sampler_student, optimizer, scheduler, name="Retrain", steps=ddim_steps, run_name=run_name)
    if compare and use_wandb:
        images, grid = util.compare_teacher_student_retrain(teacher, sampler_teacher, student, sampler_student, steps=[1, 2, 4, 8, 16, 32, 64])
        images = wandb.Image(grid, caption="left: Teacher, right: Student")
        wandb.log({"Comparison": images})
    
    del teacher, sampler_teacher, student, sampler_student, optimizer, scheduler
    torch.cuda.empty_cache()
    if use_wandb:
        wandb.finish()


def teacher_retrain_student(teacher, sampler_teacher, student, sampler_student, optimizer, scheduler, session=None, steps=20, generations=200, early_stop=True, run_name="test", cas=False):
    """
    Params: teacher, sampler_teacher, student, sampler_student, optimizer, scheduler, session=None, steps=20, generations=200, early_stop=True, run_name="test". 
    Task: trains the student model using the identical teacher model as a guide. Can be used to retrain the student into an x0 or v-prediction model.
    Not used in direct self-distillation where a teacher distills into itself.
    """
    NUM_CLASSES = 1000
    generations = 1000
    ddim_steps_teacher = steps
    TEACHER_STEPS = 1
    STUDENT_STEPS = 1
    ddim_eta = 0.0
    scale = 3.0
    updates = int(ddim_steps_teacher)
    optimizer=optimizer
    averaged_losses = []
    criterion = nn.MSELoss()
    instance = 0
    generation = 0
    all_losses = []

    with torch.no_grad():
        with teacher.ema_scope():
                sampler_teacher.make_schedule(ddim_num_steps=ddim_steps_teacher, ddim_eta=ddim_eta, verbose=False)
                sampler_student.make_schedule(ddim_num_steps=ddim_steps_teacher, ddim_eta=ddim_eta, verbose=False)
                uc = teacher.get_learned_conditioning(
                            {teacher.cond_stage_key: torch.tensor(1*[1000]).to(teacher.device)}
                            )
                with tqdm.tqdm(torch.randint(0, NUM_CLASSES, (generations,))) as tepoch:
                    for i, class_prompt in enumerate(tepoch):
                        if generation > 0 and generation % 20 == 0:
                            saving_loading.save_model(sampler_student, optimizer, scheduler, name=f"Retrain", steps=generation, run_name=run_name)

                        generation += 1
                        losses = []        
                        xc = torch.tensor([class_prompt])
                        c = teacher.get_learned_conditioning({teacher.cond_stage_key: xc.to(teacher.device)})
                        c_student = student.get_learned_conditioning({student.cond_stage_key: xc.to(student.device)})
                        
                        samples_ddim_teacher = None
                        predictions_temp = []
                        for steps in range(updates):      
                            # with autocast():    
                                instance += 1
                                samples_ddim_teacher, teacher_intermediate, x_T, pred_x0_teacher, a_t_teacher, v_teach = sampler_teacher.sample(S=TEACHER_STEPS,
                                                                conditioning=c,
                                                                batch_size=1,
                                                                shape=[3, 64, 64],
                                                                verbose=False,
                                                                x_T=samples_ddim_teacher,
                                                            
                                                                # quantize_x0 = True,
                                                                unconditional_guidance_scale=scale,
                                                                unconditional_conditioning=uc, 
                                                                eta=ddim_eta,
                                                                keep_intermediates=False,
                                                                intermediate_step = steps,
                                                                steps_per_sampling = TEACHER_STEPS,
                                                                total_steps = ddim_steps_teacher)      

                                with torch.enable_grad():
                                    with student.ema_scope():
                                        optimizer.zero_grad() 
                                        samples, pred_x0_student, st, at, v = sampler_student.sample_student(S=STUDENT_STEPS,
                                                                        conditioning=c_student,
                                                                        batch_size=1,
                                                                        shape=[3, 64, 64],
                                                                        verbose=False,
                                                                        x_T=x_T,
                                                                        
                                                                        # quantize_x0 = True,
                                                                        unconditional_guidance_scale=scale,
                                                                        unconditional_conditioning=None, 
                                                                        eta=ddim_eta,
                                                                        keep_intermediates=False,
                                                                        intermediate_step = steps,
                                                                        steps_per_sampling = STUDENT_STEPS,
                                                                        total_steps = ddim_steps_teacher)

                                        # with autocast():    
                                        #     # AUTOCAST:
                                        #     signal = at
                                        #     noise = 1 - at
                                        #     log_snr = torch.log(signal / noise)
                                        #     weight = max(log_snr, 1)
                                        #     loss = weight * criterion(pred_x0_student, pred_x0_teacher.detach())
                                        #     scaler.scale(loss).backward()
                                        #     scaler.step(optimizer)
                                        #     scaler.update()
                                        #     # torch.nn.utils.clip_grad_norm_(sampler_student.model.parameters(), 1)
                                        #     scheduler.step()
                                        #     losses.append(loss.item())

                                        # NO AUTOCAST:
                                        loss = criterion(v, v_teach) #* weight
                                        loss.backward()
                                        torch.nn.utils.clip_grad_norm_(sampler_student.model.parameters(), 1)
                                        optimizer.step()
                                        # scheduler.step()    
                                        losses.append(loss.item())
                                        
                                        
                                # if session != None and generation % 5 == 0 and generation > 0:
                                #     x_T_teacher_decode = sampler_teacher.model.decode_first_stage(pred_x0_teacher)
                                #     teacher_target = torch.clamp((x_T_teacher_decode+1.0)/2.0, min=0.0, max=1.0)
                                #     x_T_student_decode = sampler_student.model.decode_first_stage(pred_x0_student.detach())
                                #     student_target  = torch.clamp((x_T_student_decode +1.0)/2.0, min=0.0, max=1.0)
                                #     predictions_temp.append(teacher_target)
                                #     predictions_temp.append(student_target)
                        

                        if session != None and generation % 10 == 0:
                                    with torch.no_grad():
                                        images, _ = util.compare_teacher_student_retrain(teacher, sampler_teacher, student, sampler_student, steps=[256, 128, 64, 32, 16, 8, 4, 2], prompt=992)
                                        images = wandb.Image(_, caption="left: Teacher, right: Student")
                                        wandb.log({"pred_x0": images}) 
                                        sampler_student.make_schedule(ddim_num_steps=ddim_steps_teacher, ddim_eta=ddim_eta, verbose=False)
                                        sampler_teacher.make_schedule(ddim_num_steps=ddim_steps_teacher, ddim_eta=ddim_eta, verbose=False)

                        # if session != None:
                        #     with torch.no_grad():
                        #         if  generation > 0 and generation % 5 == 0 and session !=None:
                        #             img, grid = util.compare_latents(predictions_temp)
                        #             images = wandb.Image(grid, caption="left: Teacher, right: Student")
                        #             wandb.log({"Inter_Comp": images})
                        #             del img, grid, predictions_temp, x_T_student_decode, x_T_teacher_decode, student_target, teacher_target
                        #             torch.cuda.empty_cache()
                                
                            
                        
                        
                        all_losses.extend(losses)
                        # print(scheduler.get_last_lr())
                        averaged_losses.append(sum(losses) / len(losses))
                        if session != None:
                            session.log({"generation_loss":averaged_losses[-1]})
                        tepoch.set_postfix(epoch_loss=averaged_losses[-1])
                        

def distill_celeb(ddim_steps, generations, run_name, config, original_model_path, lr, start_trained=False, cas=False, compare=True, use_wandb=True):
    """
    Distill a model into a smaller model. This is done by training a student model to match the teacher model with identical initialization.
    This is not direct self-distillation as the teacher model does not distill into itself, but rather into a student model.
    """
    halvings = math.floor(math.log(ddim_steps)/math.log(2)) + 1
    updates_per_half = int(generations / halvings)

    ddim_step_list = []
    for i in range(halvings):
        ddim_step_list.append(2**i)
    ddim_step_list.reverse()
    print(f"Performing TSD for steps: {ddim_step_list}")

    for index, step in enumerate(ddim_step_list):
        steps = int(step / 2)
        model_generations = updates_per_half // steps
        config_path=config
        if index == 0 and start_trained != True:
            model_path=original_model_path
            teacher, sampler_teacher, student, sampler_student = saving_loading.create_models(config_path, model_path, student=True)
            print("Loading New Student and teacher:", step)
        else:
            model_path = f"{cwd}/data/trained_models/TSD/{run_name}/{step}.pt"
            print("Loading New Student and teacher:", step)
            teacher, sampler_teacher, optimizer, scheduler = saving_loading.load_trained(model_path, config_path)
            student = copy.deepcopy(teacher)
            sampler_student = DDIMSampler(student)
            file_path = model_path
            if not steps == 1:
                try:
                    os.remove(file_path)
                    print(f"File {file_path} has been deleted successfully")
                except FileNotFoundError:
                    print(f"Error: {file_path} not found")
                except PermissionError:
                    print(f"Error: Permission denied to delete {file_path}")
                except Exception as e:
                    print(f"An error occurred: {e}")
        
        
        if index == 0 and use_wandb:
            wandb_session = util.wandb_log(name=run_name, lr=lr, model=student, tags=["TSD"], 
            notes=f"Teacher-Student Distillation from {steps} steps with {generations} weight updates",  project="Self-Distillation")
            wandb_session.log_code(".")
    
        warmup_epochs = generations * 0.05 
        optimizer, scheduler = saving_loading.get_optimizer(sampler_student, iterations=generations, warmup_epochs=warmup_epochs, lr=lr)
        teacher_train_student_celeb(teacher, sampler_teacher, student, sampler_student, optimizer, scheduler, steps=step, generations=model_generations, 
                              early_stop=False, session=wandb_session, run_name=run_name)
        
        saving_loading.save_model(sampler_student, optimizer, scheduler, name="TSD", steps=steps, run_name=run_name)
        if compare and use_wandb:
            images, grid = util.compare_teacher_student_celeb(teacher, sampler_teacher, student, sampler_student, steps=[1, 2, 4, 8, 16, 32, 64])
            images = wandb.Image(grid, caption="left: Teacher, right: Student")
            wandb.log({"Distill Comparison": images})

        del teacher, sampler_teacher, student, sampler_student, optimizer, scheduler
        torch.cuda.empty_cache()
    if use_wandb:
        wandb.finish()
