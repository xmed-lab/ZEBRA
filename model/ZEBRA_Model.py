import random
from tqdm import tqdm
import torch.nn.functional as F
from .fmri_recon_decoder import MaskDecoder
import os
import yaml
import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .fmrienc_src.transformer_models import fMRI_Autoencoder
from timm.models.vision_transformer import Block as TransBlock
from diffusers.models.vae import Decoder
import math
from transformers import GPT2LMHeadModel



class text_MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes, bias=True, act=nn.GELU):
        super(text_MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

class TextDecoder(nn.Module):
    def __init__(self, prefix_size: int = 1280):
        super(TextDecoder, self).__init__()

        self.decoder = GPT2LMHeadModel.from_pretrained("gpt2")
        self.embedding_size = self.decoder.transformer.wte.weight.shape[1]
        self.clip_project = text_MLP((prefix_size, self.embedding_size))

    def forward(self, clip_features, gpt_tokens):
        embedding_text = self.decoder.transformer.wte(gpt_tokens)
        embedding_clip = self.clip_project(clip_features)

        embedding_clip = embedding_clip.reshape(-1, 1, self.embedding_size)
        embedding_cat = torch.cat([embedding_clip, embedding_text], dim=1)
        out = self.decoder(inputs_embeds=embedding_cat)
        return out




class BlurryReconDecoder(nn.Module):
    def __init__(self, vision_dim=1664):
        super(BlurryReconDecoder, self).__init__()
        '''Blurry Recon'''

        self.maps_projector = nn.Sequential(
            nn.Conv2d(vision_dim, 512, 1, bias=False),
            nn.GroupNorm(1, 512),
            nn.ReLU(True),
            nn.Conv2d(512, 128, 1, bias=False),
            nn.GroupNorm(1, 128),
            nn.ReLU(True),
            nn.Conv2d(128, 64, 1, bias=True),
        )
        self.bdropout = nn.Dropout(.3)
        self.bnorm = nn.GroupNorm(1, 64)
        self.bupsampler = Decoder(
            in_channels=64,
            out_channels=4,
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            block_out_channels=[32, 64, 128],
            layers_per_block=1,
        )
        self.b_maps_projector = nn.Sequential(
            nn.Conv2d(64, 512, 1, bias=False),
            nn.GroupNorm(1, 512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 1, bias=False),
            nn.GroupNorm(1, 512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 1, bias=True),
        )

    def forward(self, clip_vision_embed):
        B, N, C = clip_vision_embed.shape
        H, W = int(math.sqrt(N)), int(math.sqrt(N))
        clip_vision_embed = rearrange(clip_vision_embed, "b (h w) c -> b c h w", h=H, w=W)

        clip_vision_embed = F.interpolate(clip_vision_embed, (7, 7))

        b = self.maps_projector(clip_vision_embed)
        b = self.bdropout(b)
        b = self.bnorm(b)
        b_aux = self.b_maps_projector(b).flatten(2).permute(0, 2, 1)
        b_aux = b_aux.view(len(b_aux), 49, 512)
        b_up = self.bupsampler(b)
        return b_up, b_aux




class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output



class SubjDiscriminator(nn.Module):
    def __init__(self, feature_dim, patch_num, domain_classes):
        super(SubjDiscriminator, self).__init__()
        self.class_classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(feature_dim * patch_num, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, domain_classes),
        )
    def forward(self, x):
        y = self.class_classifier(GradReverse.apply(x))
        return y


class ImageDiscriminator(nn.Module):
    def __init__(self, embed_dim=1664, num_classes=80):
        super().__init__()
        self.attn_proj = nn.Linear(embed_dim, 1)
        self.classifier = nn.Linear(embed_dim, num_classes)
    def forward(self, x):  # x: [B, N, C]
        attn_weights = self.attn_proj(GradReverse.apply(x))
        attn_weights = torch.softmax(attn_weights, dim=1)
        x_weighted = (attn_weights * x).sum(dim=1)
        logits = self.classifier(x_weighted)
        return logits


class SubjClassifier(nn.Module):
    def __init__(self, feature_dim, patch_num, domain_classes):
        super(SubjClassifier, self).__init__()
        self.class_classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(feature_dim * patch_num, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, domain_classes),
        )
    def forward(self, x):
        y = self.class_classifier(x)
        return y



class ImageClassifier(nn.Module):
    def __init__(self, embed_dim=1664, num_classes=80):
        super().__init__()
        self.attn_proj = nn.Linear(embed_dim, 1)
        self.classifier = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        attn_weights = self.attn_proj(x)
        attn_weights = torch.softmax(attn_weights, dim=1)
        x_weighted = (attn_weights * x).sum(dim=1)
        logits = self.classifier(x_weighted)
        return logits



class CLIPProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Parameter(torch.randn(1664, 1280))
    def forward(self, x):
        x = torch.mean(x, dim=1)
        x = x @ self.proj
        return x

class Config(object):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self._yaml = f.read()
            self._dict = yaml.load(self._yaml, Loader=yaml.SafeLoader)
            self._dict['path'] = os.path.dirname(config_path)

    def __getattr__(self, name):
        if self._dict.get(name) is not None:
            return self._dict[name]

        return None

    def print(self):
        print('Model configurations:')
        print('---------------------------------')
        print(self._yaml)
        print('')
        print('---------------------------------')
        print('')


def load_fmri_backbone():
    ckpt_encoder = './pretrained_weights/fMRI2fMRI_UKB/checkpoint_120000.pth'
    cfg_file = './pretrained_weights/fMRI2fMRI_UKB/fMRI_AutoEncoder.yaml'
    config = Config(cfg_file)

    fmri_backbone = fMRI_Autoencoder(config)

    # load without module
    pretrain_metafile = torch.load(ckpt_encoder, map_location='cpu')
    model_keys = set(fmri_backbone.state_dict().keys())
    load_keys = set(pretrain_metafile['model'].keys())
    state_dict = pretrain_metafile['model']
    if model_keys != load_keys:
        print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
        if not list(model_keys)[0].startswith('module.') and list(load_keys)[0].startswith('module.'):
            state_dict = {k.replace('module.transformer.', ''): v for k, v in state_dict.items()}
    fmri_backbone.load_state_dict(state_dict, strict=True)
    print('-----------Loaded fMRI backbone-----------')
    del fmri_backbone.decoder_pos_embed
    del fmri_backbone.decoder_blocks
    del fmri_backbone.decoder_pred
    del fmri_backbone.decoder_embed
    del fmri_backbone.decoder_norm
    return fmri_backbone



class SemanticBottleneck(nn.Module):
    def __init__(self, in_dim=1024, out_dim=1664, seq_dim=256, hidden_dim=2048):
        super().__init__()

        # Token-mixing MLP: operates on sequence dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(seq_dim),
            nn.Linear(seq_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, seq_dim)
        )

        # Channel projection: map 1024 → 1664
        self.project = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):  # x: [B, 256, 1024]
        x = x.transpose(1, 2)         # [B, 1024, 256]
        x = self.mlp(x)               # [B, 1024, 256]
        x = x.transpose(1, 2)         # [B, 256, 1024]
        x = self.project(x)           # [B, 256, 1664]
        return x



class ChannelWiseAttention(nn.Module):
    def __init__(self, in_dim, num_heads=8):
        super(ChannelWiseAttention, self).__init__()

        # Project input to the attention space (through query, key, and value projections)
        self.query_proj = nn.Linear(in_dim, in_dim)
        self.key_proj = nn.Linear(in_dim, in_dim)
        self.value_proj = nn.Linear(in_dim, in_dim)

        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        # self.norm = nn.LayerNorm(in_dim)
        self.out_proj = nn.Linear(in_dim, in_dim)


    def forward(self, x):  # x: [B, N, C]
        B, N, C = x.shape

        # Query, Key, Value projections
        Q = self.query_proj(x)  # [B, N, C]
        K = self.key_proj(x)  # [B, N, C]
        V = self.value_proj(x)  # [B, N, C]

        # Scaled Dot-Product Attention (on the channel dimension)
        scores = (Q.transpose(-2, -1) @ K) / (self.head_dim ** 0.5)  # [B, H, N, N]


        # print(f"\033[92m {scores.shape} \033[0m")
        attn = torch.softmax(scores, dim=-1)  # [B, H, N, N]

        # Apply attention to the values
        attn_out = attn @ V.transpose(-2, -1)  # [B, H, N, d]
        attn_out = attn_out.transpose(-2, -1).transpose(1, 2).reshape(B, N, C)  # [B, N, C]

        # Add residual connection and layer normalization
        out = self.out_proj(attn_out)

        return out


class SparseAwareVisionProjector(nn.Module):
    def __init__(self, in_dim=1024, out_dim=1664, patch_num=256):
        super().__init__()
        self.bottleneck = SemanticBottleneck(in_dim, out_dim)
        self.broadcaster = ChannelWiseAttention(in_dim=out_dim, num_heads=1)  # 使用 Channel-wise attention

    def forward(self, x):  # x: [B, 256, 1024]
        bottleneck = self.bottleneck(x)  # [B, 1, bottleneck_dim]
        vision_embed = self.broadcaster(bottleneck)  # [B, 256, 1664]
        return vision_embed

class ZEBRA(nn.Module):
    def __init__(self):
        super(ZEBRA, self).__init__()

    def forward(self, x):
        return x







class fMRIBackbone(nn.Module):
    def __init__(self, dim, vision_dim=1664, clip_txt_emb_dim=1280, emb_dropout=0., pretrain=False):
        super().__init__()

        self.fmri_encoder = load_fmri_backbone()
        self.patch_num = 256

        self.invariant_feature_blocks = nn.ModuleList([
            TransBlock(dim, 16, mlp_ratio=4.0, qkv_bias=True,
                       drop=emb_dropout, attn_drop=emb_dropout, drop_path=emb_dropout, norm_layer=nn.LayerNorm)
            for _ in range(8)])
        self.norm_i = nn.LayerNorm(dim)

        # Shared projection head for all three components
        self.to_clip = SparseAwareVisionProjector(in_dim=dim, out_dim=vision_dim)
        self.to_clip_s = SparseAwareVisionProjector(in_dim=dim, out_dim=vision_dim)

        # Subject classifiers and discriminator
        self.subj_classifier = SubjClassifier(dim, self.patch_num, 8)
        self.subj_discriminator = SubjDiscriminator(dim, self.patch_num, 8)

        self.image_classifier = ImageClassifier(vision_dim)
        self.image_discriminator = ImageDiscriminator(vision_dim)

        # fMRI recon
        self.decoder = MaskDecoder(transformer_dim=dim)

    def forward(self, fmri, pretrain=False, infer=False):
        # fMRI encoder
        brain_embed = self.fmri_encoder.forward_encoder(fmri)  # [B, 256, 1024]

        for layer, blk in enumerate(self.invariant_feature_blocks):
            if layer == 0:
                brain_embed_i = blk(brain_embed)
            else:
                brain_embed_i = blk(brain_embed_i)
        brain_embed_i = self.norm_i(brain_embed_i)
        brain_embed_s = brain_embed - brain_embed_i  # subject-specific

        if pretrain:
            clip_embed_all = self.to_clip(brain_embed)
            if not infer:
                # Subject classification and discriminator
                pred_subj_cls = self.subj_classifier(brain_embed_s)
                pred_subj_dis = self.subj_discriminator(brain_embed_i)

                # fMRI reconstruction
                fmri_recon = self.decoder(brain_embed)
                return fmri_recon, clip_embed_all, brain_embed_i, brain_embed_s, pred_subj_cls, pred_subj_dis
            else:
                return clip_embed_all
        else:

            clip_embed = self.to_clip(brain_embed)  # invariant-specific
            clip_embed_s = self.to_clip_s(brain_embed_i)  # semantic-specific

            clip_embed_i = clip_embed - clip_embed_s

            # Reference projection (optional for reconstruction loss)
            if not infer:
                pred_image_cls = self.image_classifier(clip_embed_s)
                pred_image_dis = self.image_discriminator(clip_embed_i)
                return clip_embed, clip_embed_s, pred_image_cls, pred_image_dis
            else:
                return clip_embed_s





# for prior
from dalle2_pytorch import DiffusionPrior
from dalle2_pytorch.dalle2_pytorch import l2norm, default, exists
from dalle2_pytorch.dalle2_pytorch import RotaryEmbedding, SinusoidalPosEmb, MLP, Rearrange, repeat, rearrange, \
    prob_mask_like, LayerNorm, RelPosBias, Attention, FeedForward


class BrainDiffusionPrior(DiffusionPrior):
    """
    Differences from original:
    - Allow for passing of generators to torch random functions
    - Option to include the voxel2clip model and pass voxels into forward method
    - Return predictions when computing loss
    - Load pretrained model from @nousr trained on LAION aesthetics
    """

    def __init__(self, *args, **kwargs):
        voxel2clip = kwargs.pop('voxel2clip', None)
        super().__init__(*args, **kwargs)
        self.voxel2clip = voxel2clip

    @torch.no_grad()
    def p_sample(self, x, t, text_cond=None, self_cond=None, clip_denoised=True, cond_scale=1.,
                 generator=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=t, text_cond=text_cond,
                                                                          self_cond=self_cond,
                                                                          clip_denoised=clip_denoised,
                                                                          cond_scale=cond_scale)
        if generator is None:
            noise = torch.randn_like(x)
        else:
            noise = torch.randn_like(x)
            # noise = torch.randn(x.size(), device=x.device, dtype=x.dtype, generator=generator)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop(self, *args, timesteps=None, **kwargs):
        timesteps = default(timesteps, self.noise_scheduler.num_timesteps)
        assert timesteps <= self.noise_scheduler.num_timesteps
        is_ddim = timesteps < self.noise_scheduler.num_timesteps

        if not is_ddim:
            normalized_image_embed = self.p_sample_loop_ddpm(*args, **kwargs)
        else:
            normalized_image_embed = self.p_sample_loop_ddim(*args, **kwargs, timesteps=timesteps)

        # print("PS removed all image_embed_scale instances!")
        image_embed = normalized_image_embed  # / self.image_embed_scale
        return image_embed

    @torch.no_grad()
    def p_sample_loop_ddim(self, shape, text_cond, *, timesteps, eta=1., cond_scale=1.):
        batch, device, alphas, total_timesteps = shape[
            0], self.device, self.noise_scheduler.alphas_cumprod_prev, self.noise_scheduler.num_timesteps

        times = torch.linspace(-1., total_timesteps, steps=timesteps + 1)[:-1]

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        image_embed = torch.randn(shape, device=device)

        x_start = None  # for self-conditioning

        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        for time, time_next in time_pairs:
            alpha = alphas[time]
            alpha_next = alphas[time_next]

            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            self_cond = x_start if self.net.self_cond else None

            pred = self.net.forward_with_cond_scale(image_embed, time_cond, self_cond=self_cond, cond_scale=cond_scale,
                                                    **text_cond)

            # derive x0

            if self.predict_v:
                x_start = self.noise_scheduler.predict_start_from_v(image_embed, t=time_cond, v=pred)
            elif self.predict_x_start:
                x_start = pred
            else:
                x_start = self.noise_scheduler.predict_start_from_noise(image_embed, t=time_cond, noise=pred)

            # clip x0 before maybe predicting noise

            if not self.predict_x_start:
                x_start.clamp_(-1., 1.)

            if self.predict_x_start and self.sampling_clamp_l2norm:
                x_start = self.l2norm_clamp_embed(x_start)

            # predict noise

            pred_noise = self.noise_scheduler.predict_noise_from_start(image_embed, t=time_cond, x0=x_start)

            if time_next < 0:
                image_embed = x_start
                continue

            c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c2 = ((1 - alpha_next) - torch.square(c1)).sqrt()
            noise = torch.randn_like(image_embed) if time_next > 0 else 0.

            image_embed = x_start * alpha_next.sqrt() + \
                          c1 * noise + \
                          c2 * pred_noise

        if self.predict_x_start and self.sampling_final_clamp_l2norm:
            image_embed = self.l2norm_clamp_embed(image_embed)

        return image_embed

    @torch.no_grad()
    def p_sample_loop_ddpm(self, shape, text_cond, cond_scale=1., generator=None):
        batch, device = shape[0], self.device

        if generator is None:
            image_embed = torch.randn(shape, device=device)
        else:
            image_embed = torch.randn(shape, device=device, generator=generator)
        x_start = None  # for self-conditioning

        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        for i in tqdm(reversed(range(0, self.noise_scheduler.num_timesteps)), desc='sampling loop time step',
                      total=self.noise_scheduler.num_timesteps, disable=True):
            times = torch.full((batch,), i, device=device, dtype=torch.long)

            self_cond = x_start if self.net.self_cond else None
            image_embed, x_start = self.p_sample(image_embed, times, text_cond=text_cond, self_cond=self_cond,
                                                 cond_scale=cond_scale,
                                                 generator=generator)

        if self.sampling_final_clamp_l2norm and self.predict_x_start:
            image_embed = self.l2norm_clamp_embed(image_embed)

        return image_embed

    def p_losses(self, image_embed, times, text_cond, noise=None):
        noise = default(noise, lambda: torch.randn_like(image_embed))

        image_embed_noisy = self.noise_scheduler.q_sample(x_start=image_embed, t=times, noise=noise)

        self_cond = None
        if self.net.self_cond and random.random() < 0.5:
            with torch.no_grad():
                self_cond = self.net(image_embed_noisy, times, **text_cond).detach()

        pred = self.net(
            image_embed_noisy,
            times,
            self_cond=self_cond,
            text_cond_drop_prob=self.text_cond_drop_prob,
            image_cond_drop_prob=self.image_cond_drop_prob,
            **text_cond
        )

        if self.predict_x_start and self.training_clamp_l2norm:
            pred = self.l2norm_clamp_embed(pred)

        if self.predict_v:
            target = self.noise_scheduler.calculate_v(image_embed, times, noise)
        elif self.predict_x_start:
            target = image_embed
        else:
            target = noise

        loss = nn.functional.mse_loss(pred, target)  # mse
        # print("1", loss)
        # loss += (1 - nn.functional.cosine_similarity(pred, target).mean())
        # print("2", (1 - nn.functional.cosine_similarity(pred, target).mean()))
        return loss, pred

    def forward(
            self,
            text=None,
            image=None,
            voxel=None,
            text_embed=None,  # allow for training on preprocessed CLIP text and image embeddings
            image_embed=None,
            text_encodings=None,  # as well as CLIP text encodings
            *args,
            **kwargs
    ):
        assert exists(text) ^ exists(text_embed) ^ exists(
            voxel), 'either text, text embedding, or voxel must be supplied'
        assert exists(image) ^ exists(image_embed), 'either image or image embedding must be supplied'
        assert not (self.condition_on_text_encodings and (not exists(text_encodings) and not exists(
            text))), 'text encodings must be present if you specified you wish to condition on it on initialization'

        if exists(voxel):
            assert exists(self.voxel2clip), 'voxel2clip must be trained if you wish to pass in voxels'
            assert not exists(text_embed), 'cannot pass in both text and voxels'
            if self.voxel2clip.use_projector:
                clip_voxels_mse, clip_voxels = self.voxel2clip(voxel)
                text_embed = clip_voxels_mse
            else:
                clip_voxels = self.voxel2clip(voxel)
                text_embed = clip_voxels_mse = clip_voxels
            # text_embed = self.voxel2clip(voxel)

        if exists(image):
            image_embed, _ = self.clip.embed_image(image)

        # calculate text conditionings, based on what is passed in

        if exists(text):
            text_embed, text_encodings = self.clip.embed_text(text)

        text_cond = dict(text_embed=text_embed)

        if self.condition_on_text_encodings:
            assert exists(text_encodings), 'text encodings must be present for diffusion prior if specified'
            text_cond = {**text_cond, 'text_encodings': text_encodings}

        # timestep conditioning from ddpm

        batch, device = image_embed.shape[0], image_embed.device
        times = self.noise_scheduler.sample_random_times(batch)

        # PS: I dont think we need this? also if uncommented this does in-place global variable change
        # scale image embed (Katherine)
        # image_embed *= self.image_embed_scale

        # calculate forward loss

        loss, pred = self.p_losses(image_embed, times, text_cond=text_cond, *args, **kwargs)

        # undo the scaling so we can directly use it for real mse loss and reconstruction
        return loss, pred


class PriorNetwork(nn.Module):
    def __init__(
            self,
            dim,
            num_timesteps=None,
            num_time_embeds=1,
            # num_image_embeds = 1,
            # num_brain_embeds = 1,
            num_tokens=257,
            causal=True,
            learned_query_mode='none',
            **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.num_time_embeds = num_time_embeds
        self.continuous_embedded_time = not exists(num_timesteps)
        self.learned_query_mode = learned_query_mode

        self.to_time_embeds = nn.Sequential(
            nn.Embedding(num_timesteps, dim * num_time_embeds) if exists(num_timesteps) else nn.Sequential(
                SinusoidalPosEmb(dim), MLP(dim, dim * num_time_embeds)),
            # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange('b (n d) -> b n d', n=num_time_embeds)
        )

        if self.learned_query_mode == 'token':
            self.learned_query = nn.Parameter(torch.randn(num_tokens, dim))
        if self.learned_query_mode == 'pos_emb':
            scale = dim ** -0.5
            self.learned_query = nn.Parameter(torch.randn(num_tokens, dim) * scale)
        if self.learned_query_mode == 'all_pos_emb':
            scale = dim ** -0.5
            self.learned_query = nn.Parameter(torch.randn(num_tokens * 2 + 1, dim) * scale)
        self.causal_transformer = FlaggedCausalTransformer(dim=dim, causal=causal, **kwargs)

        self.null_brain_embeds = nn.Parameter(torch.randn(num_tokens, dim))
        self.null_image_embed = nn.Parameter(torch.randn(num_tokens, dim))

        self.num_tokens = num_tokens
        self.self_cond = False

    def forward_with_cond_scale(
            self,
            *args,
            cond_scale=1.,
            **kwargs
    ):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, brain_cond_drop_prob=1., image_cond_drop_prob=1, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
            self,
            image_embed,
            diffusion_timesteps,
            *,
            self_cond=None,
            brain_embed=None,
            text_embed=None,
            brain_cond_drop_prob=0.,
            text_cond_drop_prob=None,
            image_cond_drop_prob=0.
    ):
        if text_embed is not None:
            brain_embed = text_embed
        if text_cond_drop_prob is not None:
            brain_cond_drop_prob = text_cond_drop_prob


        # print(f"\033[92m ==== image_embed {image_embed.shape} \033[0m")
        # print(f"\033[92m ==== brain_embed {brain_embed.shape} \033[0m")

        # image_embed = image_embed.view(len(image_embed),-1,16*16)
        # text_embed = text_embed.view(len(text_embed),-1,768)
        # brain_embed = brain_embed.view(len(brain_embed),-1,16*16)
        # print(*image_embed.shape)
        # print(*image_embed.shape, image_embed.device, image_embed.dtype)

        batch, _, dim, device, dtype = *image_embed.shape, image_embed.device, image_embed.dtype
        # num_time_embeds, num_image_embeds, num_brain_embeds = self.num_time_embeds, self.num_image_embeds, self.num_brain_embeds

        # classifier free guidance masks
        brain_keep_mask = prob_mask_like((batch,), 1 - brain_cond_drop_prob, device=device)
        brain_keep_mask = rearrange(brain_keep_mask, 'b -> b 1 1')

        image_keep_mask = prob_mask_like((batch,), 1 - image_cond_drop_prob, device=device)
        image_keep_mask = rearrange(image_keep_mask, 'b -> b 1 1')

        # mask out brain embeddings with null brain embeddings

        # import pdb; pdb.set_trace()
        null_brain_embeds = self.null_brain_embeds.to(brain_embed.dtype)
        brain_embed = torch.where(
            brain_keep_mask,
            brain_embed,
            null_brain_embeds[None]
        )

        # print(f"\033[92m ==== after brain_embed {brain_embed.shape} \033[0m")


        # mask out image embeddings with null image embeddings
        null_image_embed = self.null_image_embed.to(image_embed.dtype)
        image_embed = torch.where(
            image_keep_mask,
            image_embed,
            null_image_embed[None]
        )

        # whether brain embedding is used for conditioning depends on whether brain encodings are available for attention
        # (for classifier free guidance, even though it seems from the paper it was not used in the prior ddpm, as the objective is different)
        # but let's just do it right
        if self.continuous_embedded_time:
            # if continuous cast to flat, else keep int for indexing embeddings
            diffusion_timesteps = diffusion_timesteps.type(dtype)
        time_embed = self.to_time_embeds(diffusion_timesteps)

        if self.learned_query_mode == 'token':
            learned_queries = repeat(self.learned_query, 'n d -> b n d', b=batch)
        elif self.learned_query_mode == 'pos_emb':
            pos_embs = repeat(self.learned_query, 'n d -> b n d', b=batch)
            image_embed = image_embed + pos_embs
            learned_queries = torch.empty((batch, 0, dim), device=brain_embed.device)
        elif self.learned_query_mode == 'all_pos_emb':
            pos_embs = repeat(self.learned_query, 'n d -> b n d', b=batch)
            learned_queries = torch.empty((batch, 0, dim), device=brain_embed.device)
        else:
            learned_queries = torch.empty((batch, 0, dim), device=brain_embed.device)

        tokens = torch.cat((
            brain_embed,  # 257
            time_embed,  # 1
            image_embed,  # 257
            learned_queries  # 257
        ), dim=-2)
        if self.learned_query_mode == 'all_pos_emb':
            tokens = tokens + pos_embs

        # attend
        tokens = self.causal_transformer(tokens)

        # get learned query, which should predict the image embedding (per DDPM timestep)
        pred_image_embed = tokens[..., -self.num_tokens:, :]

        return pred_image_embed


class FlaggedCausalTransformer(nn.Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            dim_head=64,
            heads=8,
            ff_mult=4,
            norm_in=False,
            norm_out=True,
            attn_dropout=0.,
            ff_dropout=0.,
            final_proj=True,
            normformer=False,
            rotary_emb=True,
            causal=True
    ):
        super().__init__()
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity()  # from latest BLOOM model and Yandex's YaLM

        self.rel_pos_bias = RelPosBias(heads=heads)

        rotary_emb = RotaryEmbedding(dim=min(32, dim_head)) if rotary_emb else None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, causal=causal, dim_head=dim_head, heads=heads, dropout=attn_dropout,
                          rotary_emb=rotary_emb),
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout, post_activation_norm=normformer)
            ]))

        self.norm = LayerNorm(dim,
                              stable=True) if norm_out else nn.Identity()  # unclear in paper whether they projected after the classic layer norm for the final denoised image embedding, or just had the transformer output it directly: plan on offering both options
        self.project_out = nn.Linear(dim, dim, bias=False) if final_proj else nn.Identity()

    def forward(self, x):
        n, device = x.shape[1], x.device

        x = self.init_norm(x)

        attn_bias = self.rel_pos_bias(n, n + 1, device=device)

        for attn, ff in self.layers:
            x = attn(x, attn_bias=attn_bias) + x
            x = ff(x) + x

        out = self.norm(x)
        return self.project_out(out)

