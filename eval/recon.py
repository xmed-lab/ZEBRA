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
from data_loader.dataset import NSD_ImageDataset

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
        "--root_dir", type=str, default='../cc2017_dataset',
    )
    parser.add_argument(
        "--weights_dir", type=str, default='../CrossSubj/pretrained_weights',
    )
    parser.add_argument(
        "--exp", type=str, default='./saved_weights',
    )
    parser.add_argument(
        "--subj", type=int, default=1, choices=[1, 2, 5, 7],
        help="Validate on which subject?",
    )
    parser.add_argument(
        "--ckpt", type=str, default='last',
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



def prepare_data(args):
    test_subs = [args.subj]
    test_dataset = NSD_ImageDataset(subjs=test_subs, image_norm=True, phase='val', val_data_fraction=1)
    test_dl = torch.utils.data.DataLoader(test_dataset, num_workers=4, batch_size=1, shuffle=False)
    return test_dl



def prepare_models(args):
    clip_seq_dim = 256
    clip_emb_dim = 1664
    clip_txt_emb_dim = 1280

    model = ZEBRA()

    model.backbone = fMRIBackbone(
                        dim = 1024,
                        vision_dim = clip_emb_dim,
                        clip_txt_emb_dim = clip_txt_emb_dim,
                        emb_dropout = 0.0
                    )

    out_dim = clip_emb_dim
    depth = 6
    dim_head = 52
    heads = clip_emb_dim//52 # heads * dim_head = clip_emb_dim
    timesteps = 100

    prior_network = PriorNetwork(
            dim=out_dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            causal=False,
            num_tokens = clip_seq_dim,
            learned_query_mode="pos_emb",
        )

    model.diffusion_prior = BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=out_dim,
        condition_on_text_encodings=False,
        timesteps=timesteps,
        cond_drop_prob=0.2,
        image_embed_scale=None,
    )
    model.clipproj = CLIPProj()
    model.blurry_recon_decoder = BlurryReconDecoder()

    if args.ckpt == "prior":
        checkpoint = torch.load(
            os.path.join(args.root_dir, f"exp_{args.exp}", f"subj_{args.subj}", "checkpoints", f"brain_model_prior.pth"),
            map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print(
            f"\033[92m Pretrained brain_model loaded from {os.path.join(args.root_dir, f'exp_{args.exp}/subj_1', 'checkpoints', f'brain_model_prior.pth')} \033[0m")
    else:
        checkpoint = torch.load(
            os.path.join(args.root_dir, f"exp_{args.exp}", f"subj_{args.subj}", "checkpoints", f"brain_model_prior_last.pth"),
            map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print(
            f"\033[92m Pretrained brain_model loaded from {os.path.join(args.root_dir, f'exp_{args.exp}/subj_1', 'checkpoints', f'brain_model_prior_last.pth')} \033[0m")

    del checkpoint

    return model




def prepare_unclip():
    # prep unCLIP
    config = OmegaConf.load("./generative_models/configs/unclip6.yaml")
    config = OmegaConf.to_container(config, resolve=True)
    unclip_params = config["model"]["params"]
    network_config = unclip_params["network_config"]
    denoiser_config = unclip_params["denoiser_config"]
    first_stage_config = unclip_params["first_stage_config"]
    conditioner_config = unclip_params["conditioner_config"]
    sampler_config = unclip_params["sampler_config"]
    scale_factor = unclip_params["scale_factor"]
    disable_first_stage_autocast = unclip_params["disable_first_stage_autocast"]

    first_stage_config['target'] = 'sgm.models.autoencoder.AutoencoderKL'
    sampler_config['params']['num_steps'] = 38

    diffusion_engine = DiffusionEngine(network_config=network_config,
                                       denoiser_config=denoiser_config,
                                       first_stage_config=first_stage_config,
                                       conditioner_config=conditioner_config,
                                       sampler_config=sampler_config,
                                       scale_factor=scale_factor,
                                       disable_first_stage_autocast=disable_first_stage_autocast)
    # set to inference
    diffusion_engine.eval().requires_grad_(False)
    diffusion_engine.to(device)

    ckpt_path = f'{args.weights_dir}/unclip6_epoch0_step110000.ckpt'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    diffusion_engine.load_state_dict(ckpt['state_dict'])
    del ckpt
    return diffusion_engine


def inference(args, model, diffusion_engine, test_dl):
    batch = {"jpg": torch.randn(1, 3, 1, 1).to(device),  # jpg doesnt get used, it's just a placeholder
             "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
             "crop_coords_top_left": torch.zeros(1, 2).to(device)}
    out = diffusion_engine.conditioner(batch)
    vector_suffix = out["vector"].to(device)
    print("vector_suffix", vector_suffix.shape)

    # get all reconstructions
    model.to(device)
    model.eval().requires_grad_(False)

    # all_images = None
    all_recons = None
    all_gts = None

    num_samples_per_image = 1
    assert num_samples_per_image == 1
    index = 0
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        for batch in tqdm(test_dl):

            fMRIs = batch["fMRIs"]  # [B, 1, 1, 256, 256]
            text = batch["txt"]
            image = batch["gt_image"]
            fMRIs = fMRIs.to(device)
            image = image.to(device)

            fMRI = fMRIs[:, :, 0]

            clip_vision_embeds = model.backbone(fMRI, infer=True)

            # Feed voxels through OpenCLIP-bigG diffusion prior
            prior_out = model.diffusion_prior.p_sample_loop(clip_vision_embeds.shape,
                                                            text_cond=dict(text_embed=clip_vision_embeds),
                                                            cond_scale=1., timesteps=20)

            prior_out = prior_out.to(device)

            # Feed diffusion prior outputs through unCLIP
            for i in range(len(fMRI)):
                index += 1
                print(index)
                gt = image[i].unsqueeze(0)
                samples = utils.unclip_recon(prior_out[[i]],
                                             diffusion_engine,
                                             vector_suffix,
                                             num_samples=num_samples_per_image,
                                             device=device)


                image_save = samples[0].permute(1, 2, 0).cpu().numpy()
                # print(f"\033[92m {image_save.shape} \033[0m")
                image_save = (image_save * 255).astype('uint8')
                image_save = cv2.resize(image_save, (224, 224), interpolation=cv2.INTER_LINEAR)
                image_save = cv2.cvtColor(image_save, cv2.COLOR_RGB2BGR)


                gt_image = gt[0].permute(1, 2, 0).cpu().numpy()
                gt_image = (gt_image * 255).astype('uint8')
                gt_image = cv2.cvtColor(gt_image, cv2.COLOR_RGB2BGR)

                image_save = np.concatenate((image_save, gt_image), axis=0)



                # 保存为 JPG
                cv2.imwrite(f"EXP/exp_{args.exp}/subj_{args.subj}/vis_img--{args.ckpt}/frame_{index}.jpg", image_save)

                if all_recons is None:
                    all_recons = samples.cpu()
                    all_gts = gt.cpu()
                else:
                    all_recons = torch.vstack((all_recons, samples.cpu()))
                    all_gts = torch.vstack((all_gts, gt.cpu()))

    return all_recons, all_gts


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

    test_dl = prepare_data(args)

    diffusion_engine = prepare_unclip()


    from model.ZEBRA_Model import (ZEBRA, fMRIBackbone, PriorNetwork, BrainDiffusionPrior, CLIPProj, BlurryReconDecoder)
    model = prepare_models(args)



    os.makedirs(f"EXP/exp_{args.exp}/subj_{args.subj}/frames_generated--{args.ckpt}", exist_ok=True)
    os.makedirs(f"EXP/exp_{args.exp}/subj_{args.subj}/vis_img--{args.ckpt}", exist_ok=True)


    all_recons, all_gts = inference(args, model, diffusion_engine, test_dl)

    # resize outputs before saving
    imsize = 256
    all_recons = transforms.Resize((imsize, imsize))(all_recons).float()
    all_gts = transforms.Resize((imsize, imsize))(all_gts).float()

    # saving
    print(f"\033[92m all_recons {all_recons.shape} \033[0m")
    print(f"\033[92m all_gts {all_gts.shape} \033[0m")

    torch.save(all_recons, f"EXP/exp_{args.exp}/subj_{args.subj}/frames_generated--{args.ckpt}/{model_name}_all_recons.pt")
    torch.save(all_gts, f"EXP/exp_{args.exp}/subj_{args.subj}/frames_generated--{args.ckpt}/{model_name}_all_gts.pt")

    if not utils.is_interactive():
        sys.exit(0)