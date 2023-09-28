import argparse
import self_distillation
import distillation
import saving_loading
import generate
import wandb
import util
import os

cwd = os.getcwd()

parser = argparse.ArgumentParser(description='Direct Self-Distillation')


parser.add_argument('--task', '-t', type=str, default= "DSDI", help='Task to perform', choices=['TSD', "DSDN", "DSDI", "DSDGL", "DSDGEXP", "SI", "SI_orig", "FID", "Inception", "NPZ", "NPZ_single", "retrain", "GET_FID"])
parser.add_argument('--model', '-m', type=str, default= "cin", help='Model type', choices=['cin', 'celeb', 'lsun_bedroom'])
parser.add_argument('--steps', '-s', type=int, default= 64, help='DDIM steps to distill from')
parser.add_argument('--updates', '-u', type=int, default= 100000, help='Number of total weight updates')
parser.add_argument('--learning_rate', '-lr', default= 0.000000002, type=float, help='Learning Rate')
parser.add_argument('--cas', '-c', type=bool, default= False, help='Include Cosine Annealing Scheduler for learning rate')
parser.add_argument('--name', '-n', type=str, help='Name to give the run, or type of run to save')
parser.add_argument('--save', '-sv', type=bool, default= True, help='Save intermediate models')
parser.add_argument('--compare', type=bool, default= True, help='Compare to original model')
parser.add_argument('--wandb', '-w', type=bool, default=False, help='Weights and Biases upload')
parser.add_argument('--cuda', '-cu', type=str, default="True", help='Cuda on/off')
parser.add_argument('--predict', '-pred', type=bool, default=False, help='either x0 or eps prediction, True = X0,  x0 uses the retrained model, eps uses the original model')
parser.add_argument('--pixels', '-p', type=int, default=256, help='256/64 pixel outputs')


if __name__ == '__main__':
    args = parser.parse_args()

    if "False" in args.cuda:
        device = 'cpu'
        print("Running on CPU")
    else:
        device = 'cuda'

    if args.pixels == 256:
        if args.model == "cin":
            if args.predict:
                model_path=f"{cwd}/models/cin256_retrained.pt"
                config_path = f"{cwd}/models/configs/cin256-v2-custom_x0.yaml"
            else:
                print("GETTING EPS MODEL")
                config_path=f"{cwd}/models/configs/cin256-v2-custom.yaml"
                model_path=f"{cwd}/models/cin256_original.ckpt"
            npz = f"{cwd}/val_saved/real_fid_both.npz"
        if args.model == "lsun_bedroom":
            if args.predict:
                model_path=f"{cwd}/models/lsun_bedrooms.ckpt"
                config_path = f"{cwd}/models/configs/lsun_bedrooms-ldm-vq-4.yaml"
            else:
                print("GETTING EPS MODEL")
                model_path=f"{cwd}/models/lsun_bedrooms.ckpt"
                config_path = f"{cwd}/models/configs/lsun_bedrooms-ldm-vq-4.yaml"
        elif args.model == "celeb":
            config_path=f"{cwd}/models/configs/celebahq-ldm-vq-4.yaml"
            model_path=f"{cwd}/models/CelebA.ckpt"
            npz = f"{cwd}/val_saved/celeb.npz"
    elif args.pixels == 64:
        print("64 model")
        if args.model == "cin":
            config_path=f"{cwd}/models/configs/cin256-v2-custom.yaml"
            model_path=f"{cwd}/models/64x64_diffusion.pt"
            npz = f"{cwd}/val_saved/real_fid_both.npz"
        elif args.model == "celeb":
            config_path=f"{cwd}/models/configs/celebahq-ldm-vq-4.yaml"
            model_path=f"{cwd}/models/CelebA.ckpt"
            npz = f"{cwd}/val_saved/celeb.npz"
   
    if args.task == "retrain":

        if args.name is None:
            args.name = f"{args.model}_retrain_{args.steps}_{args.learning_rate}_{args.updates}"
        if args.model == "cin":
            distillation.retrain(ddim_steps=args.steps, generations=args.updates, run_name=args.name, config=config_path, 
                    original_model_path=model_path, lr=args.learning_rate, start_trained=False, cas=args.cas, compare=args.compare, use_wandb=args.wandb)
        else:
            distillation.distill_celeb(ddim_steps=args.steps, generations=args.updates, run_name=args.name, config=config_path, 
                    original_model_path=model_path, lr=args.learning_rate, start_trained=False, cas=args.cas, compare=args.compare, use_wandb=args.wandb)

    if args.task == "TSD":
        if args.name is None:
            args.name = f"{args.model}_TSD_{args.predict}_{args.steps}_{args.learning_rate}_{args.updates}"
        
        if args.model == "cin":
            distillation.distill(args, config=config_path, original_model_path=model_path, start_trained=False)
        else:
            distillation.distill_celeb(ddim_steps=args.steps, generations=args.updates, run_name=args.name, config=config_path, 
                    original_model_path=model_path, lr=args.learning_rate, start_trained=False, cas=args.cas, compare=args.compare, use_wandb=args.wandb)


    elif args.task == "DSDN":
        if args.name is None:
            args.name = f"{args.model}_DSDN_{args.steps}_{args.learning_rate}_{args.updates}"

        teacher, sampler_teacher = util.create_models(config_path, model_path, student=False)
        if args.compare:
            original, sampler_original = util.create_models(config_path, model_path, student=False)

        step_scheduler = "naive"
        decrease_steps = True
        warmup_epochs = args.updates * 0.05  # The number of initial iterations to linearly increase the learning rate
        optimizer, scheduler = util.get_optimizer(sampler_teacher, iterations=args.updates, warmup_epochs=warmup_epochs, lr=args.learning_rate)
        if args.wandb:
            wandb_session = util.wandb_log(name=args.name, lr=args.learning_rate, model=teacher, tags=["DSDN"], 
                    notes=f"Direct Naive Self-Distillation from {args.steps} steps with {args.updates} weight updates",  project="Self-Distillation")
            wandb.run.log_code(".")
        else:
            wandb_session = None
        
        if args.model == "cin":
            self_distillation.self_distillation_CIN(teacher, sampler_teacher, original, sampler_original, optimizer, scheduler, session=wandb_session, 
                        steps=args.steps, gradient_updates=args.updates, run_name=args.name, step_scheduler=step_scheduler, x0=args.predict)
            
          
        elif args.model == "celeb" or args.model == "lsun_bedroom":
            self_distillation.self_distillation_CELEB(teacher, sampler_teacher, original, sampler_original, optimizer, scheduler, session=wandb_session, 
                        steps=args.steps, gradient_updates=args.updates, run_name=args.name, step_scheduler=step_scheduler)

    elif args.task == "DSDI":

        if args.name is None:
            args.name = f"{args.model}_DSDI_{args.steps}_{args.learning_rate}_{args.updates}"
        teacher, sampler_teacher = util.create_models(config_path, model_path, student=False)
        if args.compare:
            original, sampler_original = util.create_models(config_path, model_path, student=False)
        step_scheduler = "iterative"
        decrease_steps = True
        warmup_epochs = args.updates * 0.05 
        optimizer, scheduler = util.get_optimizer(sampler_teacher, iterations=args.updates, warmup_epochs=warmup_epochs, lr=args.learning_rate)
        
        if args.wandb:
            wandb_session = util.wandb_log(name=args.name, lr=args.learning_rate, model=teacher, tags=["DSDI"], 
                    notes=f"Direct Iterative Self-Distillation from {args.steps} steps with {args.updates} weight updates",  project="Self-Distillation")
            wandb.run.log_code(".")
        else:
            wandb_session = None
        
        if args.model == "cin":
            self_distillation.self_distillation_CIN(teacher, sampler_teacher, original, sampler_original, optimizer, scheduler, session=wandb_session, 
                        steps=args.steps, gradient_updates=args.updates, run_name=args.name, step_scheduler=step_scheduler, x0=args.predict)
        else:
            self_distillation.self_distillation_CELEB(teacher, sampler_teacher, original, sampler_original, optimizer, scheduler, session=wandb_session, 
                        steps=args.steps, generations=args.updates, run_name=args.name,  step_scheduler=step_scheduler)
    elif args.task == "DSDGL":

        if args.name is None:
            args.name = f"{args.model}_DSDGL_{args.steps}_{args.learning_rate}_{args.updates}"

        teacher, sampler_teacher = util.create_models(config_path, model_path, student=False)
        if args.compare:
            original, sampler_original = util.create_models(config_path, model_path, student=False)
        warmup_epochs = args.updates * 0.05 
        step_scheduler = "gradual_linear"
        decrease_steps = True
        optimizer, scheduler = util.get_optimizer(sampler_teacher, iterations=args.updates, warmup_epochs=warmup_epochs,lr=args.learning_rate)

        if args.wandb:
            wandb_session = util.wandb_log(name=args.name, lr=args.learning_rate, model=teacher, tags=["DSDGL"], 
                    notes=f"Direct Gradual Linear Self-Distillation from {args.steps} steps with {args.updates} weight updates",  project="Self-Distillation")
            wandb.run.log_code(".")
        else:
            wandb_session = None
        
        if args.model == "cin":
            self_distillation.self_distillation_CIN(teacher, sampler_teacher, original, sampler_original, optimizer, scheduler, session=wandb_session, 
                        steps=args.steps, generations=args.updates, run_name=args.name, step_scheduler=step_scheduler)
        else:
            self_distillation.self_distillation_CELEB(teacher, sampler_teacher, original, sampler_original, optimizer, scheduler, session=wandb_session, 
                        steps=args.steps, generations=args.updates, run_name=args.name, step_scheduler=step_scheduler)

    elif args.task == "DSDGEXP":

        if args.name is None:
            args.name = f"{args.model}_DSDGEXP_{args.steps}_{args.learning_rate}_{args.updates}"

        teacher, sampler_teacher = util.create_models(config_path, model_path, student=False)
        if args.compare:
            original, sampler_original = util.create_models(config_path, model_path, student=False)
        warmup_epochs = args.updates * 0.05 
        step_scheduler = "gradual_exp"
        decrease_steps = True
        optimizer, scheduler = util.get_optimizer(sampler_teacher, iterations=args.updates, warmup_epochs=warmup_epochs,lr=args.learning_rate)
        if args.wandb:
            wandb_session = util.wandb_log(name=args.name, lr=args.learning_rate, model=teacher, tags=["DSDGEXP"], 
                    notes=f"Direct Gradual Exp Self-Distillation from {args.steps} steps with {args.updates} weight updates",  project="Self-Distillation")
            wandb.run.log_code(".")
        else:
            wandb_session = None
            
        if args.model == "cin":
            self_distillation.self_distillation_CIN(teacher, sampler_teacher, original, sampler_original, optimizer, scheduler, session=wandb_session, 
                        steps=args.steps, generations=args.updates, run_name=args.name, step_scheduler=step_scheduler)
        else:
            self_distillation.self_distillation_CELEB(teacher, sampler_teacher, original, sampler_original, optimizer, scheduler, session=wandb_session, 
                        steps=args.steps, generations=args.updates, run_name=args.name, step_scheduler=step_scheduler)

    elif args.task == "SI":
        # Saves images from the trained (distilled) models saved in /data/trained_models/final_versions

        import torch
        from omegaconf import OmegaConf
        from ldm.models.diffusion.ddim import DDIMSampler
      
        if args.updates == 100000:
            print("Doing 100k, did you mean to do this? Change -u for a specific amount of generated images")
        start_path = f"{cwd}/data/trained_models/final_versions/{args.model}/"
        for train_type in os.listdir(start_path):
            if args.name != None:
                if args.name != train_type:
                    continue
            print(train_type)
            model_path = f"{start_path}{train_type}"
            model_name = f"{os.listdir(model_path)[0]}"
            model_path = f"{model_path}/{model_name}"

            config = OmegaConf.load(config_path)  
            if device == "cuda":
                ckpt = torch.load(model_path)
            else:
                ckpt = torch.load(model_path, map_location=torch.device("cpu"))
            model = saving_loading.instantiate_from_config(config.model)
            model.to(device)
            if device == "cuda":
                model.cuda()
            else:
                model.cpu()
                model.to(torch.device("cpu"))
            model.load_state_dict(ckpt["model"], strict=False)
            model.eval()
            sampler = DDIMSampler(model)
            # model, sampler, optimizer, scheduler = util.load_trained(config_path, model_path)
            if args.model == "cin":
              util.save_images(model, sampler, args.updates, train_type, [2, 4,8], verbose=True)
            else:
              saving_loading.save_images(model, sampler, args.updates, train_type, [2, 4,8], verbose=True, celeb=True)
            del model, sampler, ckpt#, optimizer, scheduler
            torch.cuda.empty_cache()

    elif args.task == "SI_orig":
        # This is a separate, although somewhat redundant, function which generates images from the original undistilled models for baseline
        # FID calculations later on.
       
        import torch
        from omegaconf import OmegaConf
        from ldm.models.diffusion.ddim import DDIMSampler
        if args.updates == 100000:
            print("Doing 100k, did you mean to do this? Change -u for a specific amount of generated images")
        if args.model == "cin":
            original, sampler_original = util.create_models(config_path, model_path, student=False)
            original.eval()
            original
            if device == "cuda":
                original.cuda()
            else:
                original.cpu()
            util.save_images(args, original, sampler_original, args.updates,"cin_original", [2,4,8, 16, 32, 64],verbose=True)
        elif args.model == "celeb":
            config_path=f"{cwd}/models/configs/celebahq-ldm-vq-4.yaml"
            model_path=f"{cwd}/models/CelebA.ckpt"
            original, sampler_original = util.create_models(config_path, model_path, student=False)
            util.save_images(args, original, sampler_original, args.updates,"celeb_original", [2,4,8, 16, 32, 64], verbose=True, celeb=True)
        elif args.model == "lsun_bedroom":
            model_path=f"{cwd}/models/lsun_bedrooms.ckpt"
            config_path = f"{cwd}/models/configs/lsun_bedrooms-ldm-vq-4.yaml"
            original, sampler_original = util.create_models(config_path, model_path, student=False)
            util.save_images(args, original, sampler_original, args.updates,"lsun_bedroom_original", [2,4,8, 16, 32, 64], verbose=True, celeb=True)
        del original, sampler_original
        torch.cuda.empty_cache()

    elif args.task == "FID":
        # Deprecated but possibly still useful. This requires a folder with locally stored reference images. Instead, you might want to opt to generate the .NPZ files
        # yourself from the original datasets using an external repository which mainly sources from torchvision dataloaders. There are many options available.
        # This also generates FID scores over the entire generated dataset, which may introduce unreasonably high estimations. For this reason, it is disabled.
       
        import pandas as pd
        import os
        import torch_fidelity
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        filename = 'metrics.csv'
        
        if not os.path.isfile(filename):
            df = pd.DataFrame({
                "model": [],
                "type" : [],
                "step": [],
                "fid": [],
                "isc": [],
                "kid": []
            })
            df.to_csv(filename, index=False)
        df = pd.read_csv(filename)
        target = "" # Fill in your target directory containing dataset images

        model = args.model
        # basic_path_source = f"{cwd}/saved_images/{model}/"
        basic_path_source = ""
        model_names = [name for name in os.listdir(basic_path_source)]
        
        for model_name in model_names:
            model_path_source = basic_path_source + f"{model_name}/"
            steps = [step for step in os.listdir(model_path_source)]
            for step in steps:
                current_path_source = model_path_source + f"{step}/"
                if  df.loc[(df['model'] == model) & (df['step'] == step) & (df['type'] == model_name)].empty:
                    try:
                        metrics = torch_fidelity.calculate_metrics(gpu=0, fid=True, isc=False, kid=False, input1=target, input2=current_path_source)
                        metrics_df = pd.DataFrame({
                        "model" : [model],
                        "type": [model_name],
                        "step": [step],
                        "fid": [metrics["frechet_inception_distance"]],
                        "isc" :[metrics["inception_score_mean"]],
                        "kid": [metrics["kernel_inception_distance_mean"]]})                            
                        df = pd.concat([df, metrics_df])
                        df.to_csv(filename, index=False)
                    except Exception as e:
                        print("Failed to create metrics for:", current_path_source)
                        print(e)
                else:
                    print("Already have metrics for:", current_path_source)

    elif args.task == "NPZ": 
        # Creates an NPZ from all the generated images for easy FID calculation
        
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        for model in ["lsun_bedroom", "CIN", "celeb"]:   
            basic_path_source = f"{cwd}/saved_images/{model}/"
            basic_path_target = f"{cwd}/NPZ/{model}"
            model_names = [name for name in os.listdir(basic_path_source)]
 
            for model_name in model_names:
                model_path_source = basic_path_source + f"{model_name}/"
                model_path_target = basic_path_target + f"_{model_name}"
                
                steps = [step for step in os.listdir(model_path_source)]
                for step in steps:
                    current_path_source = model_path_source + f"{step}/"
                    current_path_target = model_path_target + f"_{step}"
                    try:
                        util.generate_npz(current_path_source, current_path_target)
                   
                    except Exception as e:
                        print("Failed to generate npz for ", current_path_source)
                        print(e)

    elif args.task == "NPZ_single":
        # Alternative single-run for creating an npz file
        
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        current_path_source = f"{cwd}/saved_images/" # Fill in source path
        current_path_target = f"{cwd}/NPZ/"
        util.generate_npz(current_path_source, current_path_target)


    elif args.task == "GET_FID":
        # Calculates the FID score for a given model and sampler. Potentially useful for monitoring training, or comparing distillation
        # methods.
       
        import torch
        from pytorch_fid import fid_score
        for name in ["gradual_exp", "gradual_linear", "iterative", "naive", "TSD"]:
            for instance in [2, 4, 8]:
                if not os.path.exists(f"{cwd}/saved_images/FID/{name}/{instance}"):
                    os.makedirs(f"{cwd}/saved_images/FID/{name}/{instance}")
                print(f"{cwd}/saved_images/{args.model}/{name}/{instance}/")
                with torch.no_grad():
                        image_path = f"{cwd}/saved_images/{args.model}/{name}/{instance}/"
                        fid = fid_score.calculate_fid_given_paths([f"{cwd}/val_saved/imagenet.npz", 
                        image_path], batch_size = 16, device='cuda', dims=2048)
                        print(fid, name, instance)

    
    elif args.task == "Inception":
        # Calculates the Inception score for a given generated dataset (Imagenet)

        import numpy as np
        import tensorflow as tf
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from pathlib import Path
        import os
        import tqdm  
        import pandas as pd
        import torch

        filename = 'Inception_Score.csv'
        
        if not os.path.isfile(filename):
            df = pd.DataFrame({
                "type" : [],
                "step": [],
                "isc": [],
            })
            df.to_csv(filename, index=False)
        df = pd.read_csv(filename)

        batch_size = 500
        num_generated_images = 30000

        print(f"Calculating IS, expecting {num_generated_images} generating images")

        def inception_score(preds, num_splits=10):
            scores = []
            for i in range(num_splits):
                part = preds[(i * preds.shape[0] // num_splits):((i + 1) * preds.shape[0] // num_splits), :]
                kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
                kl = np.mean(np.sum(kl, 1))
                scores.append(np.exp(kl))
                
            return np.mean(scores), np.std(scores)


        def load_images_in_batches(base_folder):
            datagen = ImageDataGenerator(rescale=1./255)  # Rescale the pixel values to [0, 1]
            
            generator = datagen.flow_from_directory(
                base_folder,
                target_size=(299, 299),
                batch_size=batch_size,
                class_mode=None,  # We do not need labels, as we are only interested in predictions
                shuffle=False  # Do not shuffle to keep the order of predictions coherent
            )
            
            return generator

        model = tf.keras.applications.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
        base_folder = "" # Fill in base folders

        for name in ["gradual_exp", "gradual_linear", "iterative", "naive", "TSD", "cin_original"]:
            for instance in [2, 4, 8]:
                with torch.no_grad():
                    image_path = f"{base_folder}/{name}/{instance}/"
                    image_generator = load_images_in_batches(image_path)
                    
                    all_preds = []
                    for images_batch in image_generator:
                        preds_batch = model.predict(images_batch)
                        all_preds.extend(preds_batch)
                        print(image_generator.batch_index)
                        if image_generator.batch_index == (num_generated_images // batch_size) - 1:  
                            print("DONE!")
                            break
                print("calculating IS")            
                mean, std = inception_score(np.array(all_preds))
                print(f"{name}, {instance} steps: mean: {mean}, std: {std}")
                metrics_df = pd.DataFrame({
                    "type": [name],
                    "step": [instance],
                    "isc" :[mean]})                           
                df = pd.concat([df, metrics_df])
                df.to_csv(filename, index=False)
                del image_generator, all_preds, preds_batch
                torch.cuda.empty_cache()
