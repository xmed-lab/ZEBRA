import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from accelerate import Accelerator
import torchvision
import imageio
import cv2

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append('generative_models/')
from generative_models.sgm.models.diffusion import DiffusionEngine
from omegaconf import OmegaConf
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder, FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder2
from generative_models.sgm.util import append_dims

from tqdm import tqdm
# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# custom functions #
import utils

import warnings

warnings.filterwarnings('ignore')


def parse_arg():
    parser = argparse.ArgumentParser(description="Model Training Configuration")
    parser.add_argument(
        "--model_name", type=str, default="testing",
        help="will load ckpt for model found in ../train_logs/model_name",
    )
    parser.add_argument(
        "--data_path", type=str, default=os.getcwd(),
        help="Path to where NSD data is stored / where to download it to",
    )
    parser.add_argument(
        "--root_dir", type=str, default='../cc2017_dataset',
    )
    parser.add_argument(
        "--weights_dir", type=str, default='../CrossSubj/pretrained_weights',
    )
    parser.add_argument(
        "--exp", type=str, default='./saved_weights',
    )
    parser.add_argument(
        "--ckpt", type=str, default='',
    )
    parser.add_argument(
        "--subj", type=int, default=1, choices=[1, 2, 5, 7],
        help="Validate on which subject?",
    )
    parser.add_argument(
        "--blurry_recon", action=argparse.BooleanOptionalAction, default=False,
    )
    parser.add_argument(
        "--n_blocks", type=int, default=4,
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=4096,
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )

    args = parser.parse_args()
    return args




def prepare_models(args):
    config = OmegaConf.load("generative_models/configs/unclip6.yaml")
    config = OmegaConf.to_container(config, resolve=True)
    unclip_params = config["model"]["params"]
    sampler_config = unclip_params["sampler_config"]
    sampler_config['params']['num_steps'] = 38
    config = OmegaConf.load("generative_models/configs/inference/sd_xl_base.yaml")
    config = OmegaConf.to_container(config, resolve=True)
    refiner_params = config["model"]["params"]

    network_config = refiner_params["network_config"]
    denoiser_config = refiner_params["denoiser_config"]
    first_stage_config = refiner_params["first_stage_config"]
    conditioner_config = refiner_params["conditioner_config"]
    scale_factor = refiner_params["scale_factor"]
    disable_first_stage_autocast = refiner_params["disable_first_stage_autocast"]

    base_ckpt_path = f'{args.weights_dir}/zavychromaxl_v30.safetensors'
    base_engine = DiffusionEngine(network_config=network_config,
                                  denoiser_config=denoiser_config,
                                  first_stage_config=first_stage_config,
                                  conditioner_config=conditioner_config,
                                  sampler_config=sampler_config,  # using the one defined by the unclip
                                  scale_factor=scale_factor,
                                  disable_first_stage_autocast=disable_first_stage_autocast,
                                  ckpt_path=base_ckpt_path)
    base_engine.eval().requires_grad_(False)
    base_engine.to(device)

    print(f"\033[92m base_engine loaded \033[0m")

    base_text_embedder1 = FrozenCLIPEmbedder(
        layer=conditioner_config['params']['emb_models'][0]['params']['layer'],
        layer_idx=conditioner_config['params']['emb_models'][0]['params']['layer_idx'],
    )
    base_text_embedder1.to(device)

    print(f"\033[92m base_text_embedder1 loaded \033[0m")


    base_text_embedder2 = FrozenOpenCLIPEmbedder2(
        arch=conditioner_config['params']['emb_models'][1]['params']['arch'],
        version=conditioner_config['params']['emb_models'][1]['params']['version'],
        freeze=conditioner_config['params']['emb_models'][1]['params']['freeze'],
        layer=conditioner_config['params']['emb_models'][1]['params']['layer'],
        always_return_pooled=conditioner_config['params']['emb_models'][1]['params']['always_return_pooled'],
        legacy=conditioner_config['params']['emb_models'][1]['params']['legacy'],
    )
    base_text_embedder2.to(device)

    print(f"\033[92m base_text_embedder2 loaded \033[0m")


    batch = {"txt": "",
             "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
             "crop_coords_top_left": torch.zeros(1, 2).to(device),
             "target_size_as_tuple": torch.ones(1, 2).to(device) * 1024}
    out = base_engine.conditioner(batch)
    crossattn = out["crossattn"].to(device)
    vector_suffix = out["vector"][:, -1536:].to(device)
    print("crossattn", crossattn.shape)
    print("vector_suffix", vector_suffix.shape)
    print("---")

    batch_uc = {
        "txt": "painting, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, ugly face, distorted face, extra legs, anime",
        "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
        "crop_coords_top_left": torch.zeros(1, 2).to(device),
        "target_size_as_tuple": torch.ones(1, 2).to(device) * 1024}
    out = base_engine.conditioner(batch_uc)
    crossattn_uc = out["crossattn"].to(device)
    vector_uc = out["vector"].to(device)
    print("crossattn_uc", crossattn_uc.shape)
    print("vector_uc", vector_uc.shape)

    return base_text_embedder1, base_text_embedder2, base_engine, crossattn, vector_suffix, crossattn_uc, vector_uc





def inference(args, root_dir, model_name):


    base_text_embedder1, base_text_embedder2, base_engine, crossattn, vector_suffix, crossattn_uc, vector_uc = prepare_models(args)

    all_images = torch.load(f"{root_dir}/{model_name}_all_gts.pt")
    all_recons = torch.load(f"{root_dir}/{model_name}_all_recons.pt")  # these are the unrefined MindEye2 recons!
    # all_clipvoxels = torch.load(f"evals/{model_name}/{model_name}_all_clipvoxels.pt")
    # all_blurryrecons = torch.load(f"evals/{model_name}/{model_name}_all_blurryrecons.pt")
    all_predcaptions = torch.load(f"{root_dir}/pred_test_caption.pt")

    all_recons = transforms.Resize((768, 768))(all_recons).float()
    all_images = transforms.Resize((224, 224))(all_images).float()
    # all_blurryrecons = transforms.Resize((768, 768))(all_blurryrecons).float()

    print(all_images.shape, all_recons.shape, all_predcaptions.shape)

    num_samples = 1  # PS: I tried increasing this to 16 and picking highest cosine similarity like we did in MindEye1, it didnt seem to increase eval performance!
    img2img_timepoint = 13  # 9 # higher number means more reliance on prompt, less reliance on matching the conditioning image
    base_engine.sampler.guider.scale = 5  # 5 # cfg

    def denoiser(x, sigma, c):
        return base_engine.denoiser(base_engine.model, x, sigma, c)

    all_enhancedrecons = None
    for img_idx in tqdm(range(len(all_recons))):
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16), base_engine.ema_scope():
            base_engine.sampler.num_steps = 25

            image = all_recons[[img_idx]]
            gt = all_images[[img_idx]]


            image = image.to(device)
            prompt = all_predcaptions[[img_idx]][0]


            # z = torch.randn(num_samples,4,96,96).to(device)
            assert image.shape[-1] == 768
            z = base_engine.encode_first_stage(image * 2 - 1).repeat(num_samples, 1, 1, 1)

            openai_clip_text = base_text_embedder1(prompt)
            clip_text_tokenized, clip_text_emb = base_text_embedder2(prompt)
            clip_text_emb = torch.hstack((clip_text_emb, vector_suffix))
            clip_text_tokenized = torch.cat((openai_clip_text, clip_text_tokenized), dim=-1)
            c = {"crossattn": clip_text_tokenized.repeat(num_samples, 1, 1),
                 "vector": clip_text_emb.repeat(num_samples, 1)}
            uc = {"crossattn": crossattn_uc.repeat(num_samples, 1, 1), "vector": vector_uc.repeat(num_samples, 1)}

            noise = torch.randn_like(z)
            sigmas = base_engine.sampler.discretization(base_engine.sampler.num_steps).to(device)
            init_z = (z + noise * append_dims(sigmas[-img2img_timepoint], z.ndim)) / torch.sqrt(1.0 + sigmas[0] ** 2.0)
            sigmas = sigmas[-img2img_timepoint:].repeat(num_samples, 1)

            base_engine.sampler.num_steps = sigmas.shape[-1] - 1
            noised_z, _, _, _, c, uc = base_engine.sampler.prepare_sampling_loop(init_z, cond=c, uc=uc,
                                                                                 num_steps=base_engine.sampler.num_steps)
            for timestep in range(base_engine.sampler.num_steps):
                noised_z = base_engine.sampler.sampler_step(sigmas[:, timestep],
                                                            sigmas[:, timestep + 1],
                                                            denoiser, noised_z, cond=c, uc=uc, gamma=0)
            samples_z_base = noised_z
            samples_x = base_engine.decode_first_stage(samples_z_base)
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

            samples = samples[0]


            image_save = samples.permute(1, 2, 0).cpu().numpy()
            # print(f"\033[92m {image_save.shape} \033[0m")
            image_save = (image_save * 255).astype('uint8')
            # image_save = cv2.resize(image_save, (224, 224), interpolation=cv2.INTER_LINEAR)
            image_save = cv2.cvtColor(image_save, cv2.COLOR_RGB2BGR)

            gt_image = gt[0].permute(1, 2, 0).cpu().numpy()
            gt_image = (gt_image * 255).astype('uint8')
            gt_image = cv2.resize(gt_image, (768, 768), interpolation=cv2.INTER_LINEAR)
            gt_image = cv2.cvtColor(gt_image, cv2.COLOR_RGB2BGR)

            image_save = np.concatenate((image_save, gt_image), axis=0)

            # 保存为 JPG
            cv2.imwrite(f"{args.root_dir}/exp_{args.exp}/subj_{args.subj}/vis_img_enhance--{args.ckpt}/frame_{img_idx}.jpg", image_save)




            samples = samples.cpu()[None]
            if all_enhancedrecons is None:
                all_enhancedrecons = samples
            else:
                all_enhancedrecons = torch.vstack((all_enhancedrecons, samples))

    return all_enhancedrecons


if __name__ == "__main__":
    args = parse_arg()

    # seed all random functions
    utils.seed_everything(args.seed)

    ### Multi-GPU config ###
    local_rank = os.getenv('RANK')
    if local_rank is None:
        local_rank = 0
    else:
        local_rank = int(local_rank)
    print("LOCAL RANK ", local_rank)
    # device = accelerator.device
    device = 'cuda:0'
    print("device:", device)



    model_name = f'video_subj0{args.subj}'

    os.makedirs(f"{args.root_dir}/exp_{args.exp}/subj_{args.subj}/vis_img_enhance--{args.ckpt}", exist_ok=True)




    all_enhancedrecons = inference(args, root_dir=f"{args.root_dir}/exp_{args.exp}/subj_{args.subj}/frames_generated--{args.ckpt}", model_name=model_name)

    # resize outputs before saving
    imsize = 256
    all_enhancedrecons = transforms.Resize((imsize, imsize))(all_enhancedrecons).float()
    # saving
    print(f"\033[92m all_enhancedrecons {all_enhancedrecons.shape} \033[0m")

    torch.save(all_enhancedrecons, f"{args.root_dir}/exp_{args.exp}/subj_{args.subj}/frames_generated--{args.ckpt}/{model_name}_all_enhancedrecons.pt")
