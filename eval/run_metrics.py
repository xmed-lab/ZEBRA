import os, sys
import numpy as np
from eval_metrics import (cal_clip,
                              cal_ssim,
                              cal_alexnet,
                              cal_pixcorr,
                              cal_efficientnet,
                              cal_inceptionv3,
                              cal_swav
                              )

import argparse

import torch
from accelerate import Accelerator, DeepSpeedPlugin
# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True
import warnings
warnings.filterwarnings("ignore")




def main(data_path, model_name, mode):

    accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
    device = accelerator.device
    print("device:",device)

    all_images = torch.load(f"{data_path}/{model_name}_all_gts.pt")
    print(f"all_images {all_images.shape, all_images.max(), all_images.min()}")

    if mode == "enhance":
        all_recons = torch.load(f"{data_path}/{model_name}_all_enhancedrecons.pt")
        print(f"all_recons_enhance {all_recons.shape, all_recons.max(), all_recons.min()}")
    else:
        all_recons = torch.load(f"{data_path}/{model_name}_all_recons.pt")
        print(f"all_recons {all_recons.shape, all_recons.max(), all_recons.min()}")

    print(f"\033[92m-------- Low-level -------- \033[0m")
    ##### PixCorr
    pixcorr_score = cal_pixcorr(all_recons, all_images)
    print(f'\033[92m  {"PixCorr":<15}: {pixcorr_score:.4f} \033[0m')

    #### SSIM
    ssim_score = cal_ssim(all_recons, all_images)
    print(f'\033[92m  {"SSIM":<15}: {ssim_score:.4f} \033[0m')

    #### AlexNet
    alexnet_2_score, alexnet_5_score = cal_alexnet(all_recons, all_images)
    print(f'\033[92m  {"Alex(2)":<15}: {alexnet_2_score * 100:.2f} \033[0m')
    print(f'\033[92m  {"Alex(5)":<15}: {alexnet_5_score * 100:.2f} \033[0m')




    print(f"\033[92m-------- High-level -------- \033[0m")
    #### InceptionV3
    inceptionv3_score = cal_inceptionv3(all_recons, all_images)
    print(f'\033[92m  {"InceptionV3":<15}: {inceptionv3_score * 100:.2f} \033[0m')

    #### CLIP
    clip_score = cal_clip(all_recons, all_images)
    print(f'\033[92m  {"CLIP":<15}: {clip_score * 100:.2f} \033[0m')

    #### Efficient Net
    efficientnet_score = cal_efficientnet(all_recons, all_images)
    print(f'\033[92m  {"EfficientNet":<15}: {efficientnet_score:.3f} \033[0m')

    #### SwAV
    swav_score = cal_swav(all_recons, all_images)
    print(f'\033[92m  {"SwAV":<15}: {swav_score:.3f} \033[0m')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, default="/codes/NeuroClips/Animatediff/StableDiffusion", )
    parser.add_argument("--inference-config", type=str, default="configs/inference/inference-v1.yaml")
    parser.add_argument(
        "--root_dir", type=str, default='/data/cc2017_dataset',
    )
    parser.add_argument(
        "--exp", type=str, default='', required=True
    )
    parser.add_argument(
        "--mode", type=str, default='', required=False
    )
    parser.add_argument(
        "--ckpt", type=str, default='', required=True
    )
    parser.add_argument(
        "--subj", type=int, default=1, choices=[1, 2, 5, 7],
        help="Validate on which subject?",
    )
    args = parser.parse_args()


    model_name = f'video_subj0{args.subj}'

    data_path = f"{args.root_dir}/exp_{args.exp}/subj_{args.subj}/frames_generated--{args.ckpt}"


    print(f"\033[92m Evaluating results from: {data_path} \033[0m")

    main(data_path=data_path, model_name=model_name, mode=args.mode)