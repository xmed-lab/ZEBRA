import os
import sys
sys.path.append('generative_models/')
import argparse
import numpy as np
from tqdm import tqdm
import gc
import wandb
import inspect
import torch
import torch.nn as nn
from accelerate import Accelerator
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder, FrozenOpenCLIPEmbedder2 # bigG embedder from OpenCLIP
from model.ZEBRA_Model import (ZEBRA, CLIPProj, fMRIBackbone, PriorNetwork, BrainDiffusionPrior, BlurryReconDecoder)
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import utils
from einops import rearrange, repeat
from diffusers import AutoencoderKL
from data_loader.dataset import NSD_ImageDataset


def kl_divergence_loss(p, q):
    p = F.log_softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    return F.kl_div(p, q, reduction='batchmean')


def orthogonal_loss(feat1, feat2):
    """
    Computes orthogonal loss with feature normalization
    """
    # Normalize features
    feat1_norm = F.normalize(feat1, dim=1)
    feat2_norm = F.normalize(feat2, dim=1)

    # Compute cosine similarity
    cos_sim = torch.abs(torch.sum(feat1_norm * feat2_norm, dim=1))

    # Square and mean across batch
    loss = torch.mean(cos_sim)
    return loss



def save_ckpt(tag, epoch, model, optimizer, lr_scheduler, losses, test_losses, lrs):
    ckpt_path = outdir+f'/{tag}.pth'
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save({
            'epoch': epoch,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'train_losses': losses,
            'test_losses': test_losses,
            'lrs': lrs,
            }, ckpt_path)
    print(f"---saved {outdir}/{tag} ckpt!---")




def calculate_category_accuracy(predicted_labels, true_labels):
    num_classes = true_labels.shape[1]
    accuracies = []

    for i in range(num_classes):
        correct_predictions = np.sum(np.logical_and(true_labels[:, i] == 1, predicted_labels[:, i] == 1))
        total_samples = np.sum(true_labels[:, i] == 1)
        if total_samples > 0:
            accuracy = correct_predictions / total_samples
        else:
            accuracy = 0
        accuracies.append(accuracy)

    return accuracies

def prepare_data(args):

    train_subs = np.arange(1, 9)
    train_nsd_subs = np.arange(1, 9)
    train_nsd_subs = train_nsd_subs[train_nsd_subs != args.subj]

    val_subs = [args.subj]


    train_dataset = NSD_ImageDataset(subjs=train_subs, subjs_nsd=train_nsd_subs, image_norm=True, phase='train')
    val_dataset = NSD_ImageDataset(subjs=val_subs, image_norm=True, phase='val', val_data_fraction=1)
    train_dl = torch.utils.data.DataLoader(train_dataset, num_workers=4, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_dl = torch.utils.data.DataLoader(val_dataset, num_workers=4, batch_size=20, shuffle=False)


    return train_dl, test_dl



def prepare_models(args):
    clip_img_embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        output_tokens=True,
        only_tokens=True,
        cache_dir=args.cache_dir
    )
    clip_img_embedder.to(device)

    clip_seq_dim = 256
    clip_emb_dim = 1664
    clip_txt_emb_dim = 1280

    model = ZEBRA()




    autoenc, cnx = None, None
    if not args.pretrain:

        model.backbone = fMRIBackbone(
            dim=1024,
            vision_dim=clip_emb_dim,
            clip_txt_emb_dim=clip_txt_emb_dim,
            emb_dropout=0.25,
            pretrain=False
        )

        model.clipproj = CLIPProj()

        print(f"\033[91m >>> Loading ZEBRA Backbone weights >>> \033[0m")
        checkpoint = torch.load(f"{args.exp_dir}/checkpoints/brain_model_last.pth", map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        del checkpoint
        print(f"\033[92m <<< ZEBRA Backbone weights loaded from {args.exp_dir}/checkpoints/brain_model_last.pth <<< \033[0m")

        model.blurry_recon_decoder = BlurryReconDecoder()



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

        autoenc = AutoencoderKL(
            down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
            up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            sample_size=256,
        )
        ckpt = torch.load(f'{args.cache_dir}sd_image_var_autoenc.pth')
        autoenc.load_state_dict(ckpt)

        autoenc.eval()
        autoenc.requires_grad_(False)
        autoenc.to(device)

        from autoencoder.convnext import ConvnextXL
        cnx = ConvnextXL(f'{args.cache_dir}convnext_xlarge_alpha0.75_fullckpt.pth')
        cnx.requires_grad_(False)
        cnx.eval()
        cnx.to(device)

        model.blurry_recon_decoder = BlurryReconDecoder()

    else:

        model.backbone = fMRIBackbone(
            dim=1024,
            vision_dim=clip_emb_dim,
            clip_txt_emb_dim=clip_txt_emb_dim,
            emb_dropout=0.1,
            pretrain=True
        )

        model.clipproj = CLIPProj()
        checkpoint = torch.load(f'{args.cache_dir}/coco_tokens_avg_proj.pth')
        model.clipproj.load_state_dict(checkpoint)


    utils.count_params(model)

    clip_txt_embedder = FrozenOpenCLIPEmbedder2(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        layer="last",
        legacy=False,
        always_return_pooled=True,
        cache_dir=args.cache_dir

    )
    clip_txt_embedder.to(device)



    if args.pretrain:
        for param in model.parameters():
            param.requires_grad_(True)
    else:
        for param in model.parameters():
            param.requires_grad_(True)
        for param in model.clipproj.parameters():
            param.requires_grad_(True)

        frozen_modules = [
            model.backbone.fmri_encoder,
            model.backbone.invariant_feature_blocks,
            model.backbone.norm_i,
            model.backbone.subj_classifier,
            model.backbone.subj_discriminator,
            model.backbone.decoder
        ]
        for module in frozen_modules:
            for param in module.parameters():
                param.requires_grad_(False)

    return model, clip_img_embedder, clip_txt_embedder, autoenc, cnx


def trainable_modules_check(is_main_process, model):
    if is_main_process:
        print(f"\033[92m================================== \033[0m")
        print(f"\033[92m Checking ... \033[0m")
        print(f"\033[92m================================== \033[0m")
        for name, param in model.named_parameters():
            if param.requires_grad == False:
                print(f"\033[94m Frozen: {name} \033[0m")
            else:
                print(f"\033[91m Trainable: {name} \033[0m")



def train(args):

    train_dl, test_dl = prepare_data(args)
    model, clip_img_embedder, clip_txt_embedder, autoenc, cnx = prepare_models(args)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.max_lr)

    num_iterations_per_epoch = len(train_dl)

    print("batch_size =", args.batch_size, "num_iterations_per_epoch =", num_iterations_per_epoch)

    if args.lr_scheduler_type == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=int(np.floor(args.num_epochs*num_iterations_per_epoch)),
            last_epoch=-1
        )
    elif args.lr_scheduler_type == 'cycle':
        total_steps=int(np.floor(args.num_epochs*num_iterations_per_epoch))
        print("total_steps", total_steps)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.max_lr,
            total_steps=total_steps,
            final_div_factor=1,
            div_factor=10,
            last_epoch=-1, pct_start=0.1
        )
    else:
        total_steps = int(np.floor(args.num_epochs * num_iterations_per_epoch))
        print("total_steps", total_steps)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=2, T_mult=2
        )

    epoch = 0
    losses, test_losses, lrs = [], [], []
    best_metric = 0
    torch.cuda.empty_cache()
    model, optimizer, train_dl, lr_scheduler = accelerator.prepare(model, optimizer, train_dl, lr_scheduler)
    loss_cls = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    global_step = 0


    if num_devices > 1 and distributed:
        model = model.module


    trainable_modules_check(accelerator.is_main_process, model)


    if args.resume_from_ckpt is not None:
        checkpoint = torch.load(args.resume_from_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        epoch = checkpoint['epoch'] + 1
        print(f"\033[92m ************ Load from checkpoint at epoch {epoch} \033[0m")
        del checkpoint



    for epoch in tqdm(range(epoch, args.num_epochs), disable=(local_rank!=0)):
        model.train()

        for idx, batch in enumerate(tqdm(train_dl, disable=(local_rank!=0))):

            with torch.cuda.amp.autocast(dtype=data_type):
                optimizer.zero_grad()

                fMRIs = batch["fMRIs"] # [B, 1, 256, 256]
                subj_lbl = batch["subj_lbl"]
                image = batch["gt_image"]
                text = batch["txt"]
                cls_labels = batch["multi_cls_labels"]

                fMRIs = fMRIs.to(device)
                subj_lbl = subj_lbl.to(device).long()
                image = image.to(device).float()

                clip_vision_target = clip_img_embedder(image)

                if args.pretrain:
                    fMRIs, perm, betas, select = utils.mixco(fMRIs)

                    fmri_recon, clip_embed_all, brain_embed_i, brain_embed_s, pred_subj_cls, pred_subj_dis = model.backbone(fMRIs, pretrain=True)




                    '''============ Brain Recon ============'''
                    loss_recon = F.mse_loss(
                        rearrange(fmri_recon, 'b c h w -> b (c h w)'),
                        rearrange(fMRIs, 'b c h w -> b (c h w)')
                    )

                    '''============ Vision Embeds Align ============'''
                    clip_vision_target_norm = nn.functional.normalize(clip_vision_target.flatten(1), dim=-1)

                    clip_vision_embeds_norm = nn.functional.normalize(clip_embed_all.flatten(1), dim=-1)  # [B, C * H * W]
                    loss_clip_vision = utils.mixco_nce(
                        clip_vision_embeds_norm,
                        clip_vision_target_norm,
                        temp=.006,
                        perm=perm, betas=betas, select=select
                    )


                    '''============ Subject Label Prediction ============'''
                    loss_subj_cls = F.cross_entropy(
                        pred_subj_cls,
                        subj_lbl.long()
                    )
                    loss_subj_dis = F.cross_entropy(
                        pred_subj_dis,
                        subj_lbl.long()
                    )
                    '''============ Overall Loss ============'''
                    loss = loss_recon \
                           + loss_clip_vision \
                           + loss_subj_cls + loss_subj_dis

                else:
                    clip_embed, clip_embed_s, pred_image_cls, pred_image_dis = model.backbone(fMRIs)

                    '''============ Prior Train ============'''
                    loss_prior, prior_out = model.diffusion_prior(text_embed=clip_embed_s,
                                                                  image_embed=clip_vision_target)

                    '''============ Vision Embeds Align ============'''
                    clip_vision_target_norm = nn.functional.normalize(clip_vision_target.flatten(1), dim=-1)

                    clip_vision_embeds_norm = nn.functional.normalize(clip_embed.flatten(1), dim=-1)  # [B, C * H * W]
                    loss_clip_vision = utils.mixco_nce(
                        clip_vision_embeds_norm,
                        clip_vision_target_norm)


                    clip_vision_embeds_norm_s = nn.functional.normalize(clip_embed_s.flatten(1), dim=-1)  # [B, C * H * W]
                    loss_clip_vision_s = utils.mixco_nce(
                        clip_vision_embeds_norm_s,
                        clip_vision_target_norm)


                    '''============ Image Label Prediction ============'''
                    pred_image_cls_all = model.backbone.image_classifier(clip_embed)
                    loss_image_cls_all = loss_cls(
                        pred_image_cls_all,
                        cls_labels.float()
                    )

                    loss_image_cls = loss_cls(
                        pred_image_cls,
                        cls_labels.float()
                    )

                    loss_image_dis = loss_cls(
                        pred_image_dis,
                        cls_labels.float()
                    )

                    '''============ Text Embeds Align ============'''
                    clip_text_embeds = model.clipproj(clip_embed)
                    _, clip_text_target = clip_txt_embedder(text)
                    clip_text_target_norm = nn.functional.normalize(clip_text_target.flatten(1), dim=-1)
                    clip_text_embeds_norm = F.normalize(clip_text_embeds.flatten(1), dim=-1)  # [B * S, C]
                    loss_clip_txt = utils.mixco_nce(clip_text_embeds_norm, clip_text_target_norm)



                    '''============ Blurry Recon ============'''
                    image_enc_pred, transformer_feats = model.blurry_recon_decoder(prior_out)
                    image_enc = autoenc.encode(2 * image - 1).latent_dist.mode() * 0.18215
                    loss_blurry = l1(image_enc_pred, image_enc)


                    '''============ Overall Loss ============'''
                    loss = loss_prior * 30 + loss_clip_vision * 0.5 + loss_clip_vision_s + loss_image_cls_all * 0.5 + loss_image_cls + loss_image_dis + loss_clip_txt * 0.25 + loss_blurry * 0.5



                utils.check_loss(loss)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                losses.append(loss.item())
                lrs.append(optimizer.param_groups[0]['lr'])

                if args.lr_scheduler_type is not None:
                    lr_scheduler.step()
                global_step += 1

                if args.use_wandb and accelerator.is_main_process:
                    wandb.log({"lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                    wandb.log({"loss": loss.item()}, step=global_step)
                    if args.pretrain:
                        wandb.log({"loss_recon": loss_recon.item()}, step=global_step)
                        wandb.log({"loss_clip_vision": loss_clip_vision.item()}, step=global_step)
                        wandb.log({"loss_subj_cls": loss_subj_cls.item()}, step=global_step)
                        wandb.log({"loss_subj_dis": loss_subj_dis.item()}, step=global_step)
                    else:
                        wandb.log({"loss_prior": loss_prior.item()}, step=global_step)
                        wandb.log({"loss_clip_vision": loss_clip_vision.item()}, step=global_step)
                        wandb.log({"loss_clip_vision_s": loss_clip_vision_s.item()}, step=global_step)
                        wandb.log({"loss_clip_txt": loss_clip_txt.item()}, step=global_step)
                        wandb.log({"loss_image_cls": loss_image_cls.item()}, step=global_step)
                        wandb.log({"loss_image_dis": loss_image_dis.item()}, step=global_step)
                        wandb.log({"loss_blurry": loss_blurry.item()}, step=global_step)





        # ==================================================================================
        # Test begin
        # ==================================================================================
        model.eval()

        test_fwd_percent_correct = []
        test_bwd_percent_correct = []
        text_fwd_percent_correct = []
        test_blurry_pixcorr = []


        if accelerator.is_main_process:
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=data_type):
                for test_i, batch in enumerate(test_dl):

                    fMRIs = batch["fMRIs"]  # [B, 1, 1, 256, 256]
                    text = batch["txt"]
                    image = batch["gt_image"]
                    fMRIs = fMRIs.to(device)
                    image = image.to(device)
                    cls_labels = batch["multi_cls_labels"]

                    fMRI = fMRIs[:, :, 0]

                    clip_vision_target = clip_img_embedder(image.float())
                    _, clip_text_target = clip_txt_embedder(text)


                    if args.pretrain:
                        clip_vision_embeds = model.backbone(fMRI, pretrain=True, infer=True)


                        clip_text_embeds = model.clipproj(clip_vision_embeds)
                        clip_text_norm = nn.functional.normalize(clip_text_embeds.flatten(1), dim=-1)
                        clip_vision_norm = nn.functional.normalize(clip_vision_embeds.flatten(1), dim=-1)



                    else:
                        clip_vision_embeds = model.backbone(fMRI, infer=True)

                        _, prior_out = model.diffusion_prior(text_embed=clip_vision_embeds,
                                                             image_embed=clip_vision_target)
                        clip_text_embeds = model.clipproj(prior_out)
                        clip_text_norm = nn.functional.normalize(clip_text_embeds.flatten(1), dim=-1)
                        clip_vision_norm = nn.functional.normalize(prior_out.flatten(1), dim=-1)


                        random_samps = np.random.choice(np.arange(len(image)), size=len(image) // 4 + 1, replace=False)
                        image_enc_pred, _ = model.blurry_recon_decoder(prior_out[random_samps])
                        blurry_recon_images = (
                                autoenc.decode(image_enc_pred / 0.18215).sample / 2 + 0.5).clamp(0, 1)
                        pixcorr = utils.pixcorr(image[random_samps], blurry_recon_images)
                        test_blurry_pixcorr.append(pixcorr.item())



                    clip_vision_target_norm = nn.functional.normalize(clip_vision_target.flatten(1), dim=-1)
                    clip_text_target_norm = nn.functional.normalize(clip_text_target.flatten(1), dim=-1)


                    labels = torch.arange(len(clip_text_norm)).to(clip_text_norm.device)
                    text_fwd_percent_correct.append(
                        utils.topk(utils.batchwise_cosine_similarity(clip_text_norm, clip_text_target_norm), labels,
                                   k=5).item())

                    # forward and backward top 1 accuracy
                    labels = torch.arange(len(clip_vision_norm)).to(clip_vision_norm.device)
                    test_fwd_percent_correct.append(utils.topk(
                        utils.batchwise_cosine_similarity(clip_vision_norm, clip_vision_target_norm), labels, k=1).item())
                    test_bwd_percent_correct.append(utils.topk(
                        utils.batchwise_cosine_similarity(clip_vision_target_norm, clip_vision_norm), labels, k=1).item())


                print(f'\033[92m Evaluating Epoch {epoch} ... \033[0m')
                print(f'\033[92m \ttest_fwd_percent_correct: {np.mean(test_fwd_percent_correct)} \033[0m')
                print(f'\033[92m \ttest_bwd_percent_correct: {np.mean(test_bwd_percent_correct)} \033[0m')
                print(f'\033[92m \ttext_fwd_percent_correct: {np.mean(text_fwd_percent_correct)} \033[0m')
                if not args.pretrain:
                    print(f'\033[92m \ttest_blurry_pixcorr     : {np.mean(test_blurry_pixcorr)} \033[0m')

                if args.use_wandb:
                    wandb.log({"test_fwd_percent_correct": np.mean(test_fwd_percent_correct)}, step=global_step)
                    wandb.log({"test_bwd_percent_correct": np.mean(test_bwd_percent_correct)}, step=global_step)
                    wandb.log({"text_fwd_percent_correct": np.mean(text_fwd_percent_correct)}, step=global_step)
                    if not args.pretrain:
                        wandb.log({"test_blurry_pixcorr": np.mean(test_blurry_pixcorr)}, step=global_step)

            metric = np.mean(test_fwd_percent_correct) + np.mean(test_bwd_percent_correct) + np.mean(text_fwd_percent_correct)

            # Save model checkpoint and reconstruct
            if metric > best_metric:
                best_metric = metric
                best_epoch = epoch
                print(f"\033[92m New best test metric: {best_metric} \033[0m")
                if args.pretrain:
                    save_ckpt(f'brain_model', epoch, model, optimizer, lr_scheduler, losses, test_losses, lrs)
                else:
                    save_ckpt(f'brain_model_prior', epoch, model, optimizer, lr_scheduler, losses, test_losses, lrs)

            else:
                print(f"\033[91m Current metric: {metric}, best metric loss is {best_metric} in Epoch {best_epoch} \033[0m")

        # wait for other GPUs to catch up if needed
        accelerator.wait_for_everyone()
        torch.cuda.empty_cache()
        gc.collect()

    if args.ckpt_saving:
        if args.pretrain:
            save_ckpt(f'brain_model_last', epoch, model, optimizer, lr_scheduler, losses, test_losses, lrs)
        else:
            save_ckpt(f'brain_model_prior_last', epoch, model, optimizer, lr_scheduler, losses, test_losses, lrs)
    print("\n===Finished!===\n")


if __name__ == "__main__":
    ### Multi-GPU config ###
    local_rank = os.getenv('RANK')
    if local_rank is None:
        local_rank = 0
    else:
        local_rank = int(local_rank)
    print("LOCAL RANK ", local_rank)

    data_type = torch.float16  # change depending on your mixed_precision
    num_devices = torch.cuda.device_count()
    if num_devices == 0: num_devices = 1

    # First use "accelerate config" in terminal and setup using deepspeed stage 2 with CPU offloading!
    accelerator = Accelerator(split_batches=False, mixed_precision="fp16")

    print("PID of this process =", os.getpid())
    device = accelerator.device
    # device = 'cuda:0'
    print("device:", device)
    world_size = accelerator.state.num_processes
    distributed = not accelerator.state.distributed_type == 'NO'
    num_devices = torch.cuda.device_count()
    if num_devices == 0 or not distributed: num_devices = 1
    num_workers = num_devices
    print(accelerator.state)

    print("distributed =", distributed, "num_devices =", num_devices, "local rank =", local_rank, "world size =",
          world_size, "data_type =", data_type)
    print = accelerator.print  # only print if local_rank=0

    parser = argparse.ArgumentParser(description="Model Training Configuration")
    parser.add_argument(
        "--subj", type=int, default=1, choices=[1, 2, 5, 7],
        help="Validate on which subject?",
    )
    parser.add_argument(
        "--batch_size", type=int, default=10,
        help="Batch size can be increased by 10x if only training retreival submodule and not diffusion prior",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=150,
        help="number of epochs of training",
    )
    parser.add_argument(
        "--lr_scheduler_type", type=str, default='cycle', choices=['cycle', 'linear', 'cosine'],
    )
    parser.add_argument(
        "--exp_dir", type=str, default='',
    )
    parser.add_argument(
        "--ckpt_saving", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument("--pretrained-model-path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument(
        "--cache_dir", type=str, default='./pretrained_weights/',
    )
    parser.add_argument(
        "--pretrain", action=argparse.BooleanOptionalAction, default=False,
        help="whether to train diffusion prior (True) or just rely on retrieval part of the pipeline (False)",
    )
    parser.add_argument(
        "--resume_from_ckpt", type=str, default=None,
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--max_lr", type=float, default=3e-4,
    )
    parser.add_argument(
        "--use_wandb",  default=False,
    )
    args = parser.parse_args()

    # seed all random functions
    utils.seed_everything(args.seed)

    os.makedirs(f'{args.exp_dir}/checkpoints/', exist_ok=True)
    outdir = os.path.abspath(f'{args.exp_dir}/checkpoints')

    if args.use_wandb and accelerator.is_main_process:
        *_, config = inspect.getargvalues(inspect.currentframe())
        if args.pretrain:
            wandb.init(project="ZEBRA", name=f"pretrain--exp_{args.exp_dir.split('exp_')[-1]}")
        else:
            wandb.init(project="ZEBRA", name=f"prior--exp_{args.exp_dir.split('exp_')[-1]}")
    train(args)



