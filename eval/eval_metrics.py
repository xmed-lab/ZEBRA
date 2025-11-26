import torch
import numpy as np
from torchvision import transforms
from torchvision.models import alexnet, AlexNet_Weights
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import clip
import scipy as sp
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

@torch.no_grad()
def two_way_identification(all_recons, all_images, model, preprocess, feature_layer=None, return_avg=True, device='cuda'):
    preds = model(torch.stack([preprocess(recon) for recon in all_recons], dim=0).to(device))
    reals = model(torch.stack([preprocess(indiv) for indiv in all_images], dim=0).to(device))
    if feature_layer is None:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).cpu().numpy()

    r = np.corrcoef(reals, preds)
    r = r[:len(all_images), len(all_images):]
    congruents = np.diag(r)

    success = r < congruents
    success_cnt = np.sum(success, 0)

    if return_avg:
        perf = np.mean(success_cnt) / (len(all_images)-1)
        return perf
    else:
        return success_cnt, len(all_images)-1



def cal_pixcorr(all_recons, all_images):
    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
    ])

    # Flatten images while keeping the batch dimension
    all_images_flattened = preprocess(all_images).reshape(len(all_images), -1).cpu()
    all_recons_flattened = preprocess(all_recons).reshape(len(all_recons), -1).cpu()

    # print(all_images_flattened.shape)
    # print(all_recons_flattened.shape)

    corrsum = 0
    for i in range(len(all_images)):
        if all_recons_flattened[i].max() == 0:
            continue
        corrsum += np.corrcoef(all_images_flattened[i], all_recons_flattened[i])[0][1]
        # print(f"\033[92m {i+1, np.corrcoef(all_images_flattened[i], all_recons_flattened[i])[0][1]} \033[0m")
    corrmean = corrsum / len(all_images)
    pixcorr = corrmean
    return pixcorr


def cal_ssim(all_recons, all_images):
    # see https://github.com/zijin-gu/meshconv-decoding/issues/3
    from skimage.color import rgb2gray
    from skimage.metrics import structural_similarity as ssim

    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
    ])

    # print(f"\033[92m all_images {all_images.shape} \033[0m")
    # print(f"\033[92m all_recons {all_recons.shape} \033[0m")


    # convert image to grayscale with rgb2grey
    img_gray = rgb2gray(preprocess(all_images).permute((0, 2, 3, 1)).cpu())
    recon_gray = rgb2gray(preprocess(all_recons).permute((0, 2, 3, 1)).cpu())
    # print("converted, now calculating ssim...")

    # img_gray = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min())

    # print(f"\033[92m img_gray {img_gray.shape, img_gray.max(), img_gray.min()} \033[0m")
    # print(f"\033[92m recon_gray {recon_gray.shape, recon_gray.max(), recon_gray.min()} \033[0m")


    ssim_score = []
    for im, rec in zip(img_gray, recon_gray):
        ssim_score.append(
            ssim(rec, im, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                 data_range=1.0))
    ssim = np.mean(ssim_score)
    return ssim


def cal_alexnet(all_recons, all_images, device="cuda"):
    alex_weights = AlexNet_Weights.IMAGENET1K_V1

    alex_model = create_feature_extractor(alexnet(weights=alex_weights), return_nodes=['features.4', 'features.11']).to(
        device)
    alex_model.eval().requires_grad_(False)

    # see alex_weights.transforms()
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # layer = 'early, AlexNet(2)'
    # print(f"\n---{layer}---")
    all_per_correct = two_way_identification(all_recons.to(device).float(), all_images,
                                             alex_model, preprocess, 'features.4')
    alexnet2 = np.mean(all_per_correct)
    # print(f"2-way Percent Correct: {alexnet2:.4f}")

    # layer = 'mid, AlexNet(5)'
    # print(f"\n---{layer}---")
    all_per_correct = two_way_identification(all_recons.to(device).float(), all_images,
                                             alex_model, preprocess, 'features.11')
    alexnet5 = np.mean(all_per_correct)
    # print(f"2-way Percent Correct: {alexnet5:.4f}")

    return alexnet2, alexnet5


def cal_inceptionv3(all_recons, all_images, device="cuda"):

    weights = Inception_V3_Weights.DEFAULT
    inception_model = create_feature_extractor(inception_v3(weights=weights),
                                               return_nodes=['avgpool']).to(device)
    inception_model.eval().requires_grad_(False)

    # see weights.transforms()
    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    all_per_correct = two_way_identification(all_recons, all_images,
                                             inception_model, preprocess, 'avgpool')

    inception_score = np.mean(all_per_correct)
    # print(f"2-way Percent Correct: {inception:.4f}")

    return inception_score


def cal_clip(all_recons, all_images, device="cuda"):
    clip_model, preprocess = clip.load("ViT-L/14", device=device)

    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    all_per_correct = two_way_identification(all_recons, all_images,
                                             clip_model.encode_image, preprocess, None)  # final layer
    clip_ = np.mean(all_per_correct)
    # print(f"2-way Percent Correct: {clip_:.4f}")
    return clip_


def cal_efficientnet(all_recons, all_images, device="cuda"):

    weights = EfficientNet_B1_Weights.DEFAULT
    eff_model = create_feature_extractor(efficientnet_b1(weights=weights),
                                         return_nodes=['avgpool'])
    eff_model.eval().requires_grad_(False)

    # see weights.transforms()
    preprocess = transforms.Compose([
        transforms.Resize(432, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    gt = eff_model(preprocess(all_images))['avgpool']
    gt = gt.reshape(len(gt), -1).cpu().numpy()
    fake = eff_model(preprocess(all_recons))['avgpool']
    fake = fake.reshape(len(fake), -1).cpu().numpy()

    effnet = np.array([sp.spatial.distance.correlation(gt[i], fake[i]) for i in range(len(gt))]).mean()
    # print("Distance:", effnet)

    return effnet


def cal_swav(all_recons, all_images):
    swav_model = torch.hub.load('facebookresearch/swav:main', 'resnet50', verbose=False)
    swav_model = create_feature_extractor(swav_model,
                                          return_nodes=['avgpool'])
    swav_model.eval().requires_grad_(False)

    preprocess = transforms.Compose([
        transforms.Resize(240, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    gt = swav_model(preprocess(all_images))['avgpool']
    gt = gt.reshape(len(gt), -1).cpu().numpy()
    fake = swav_model(preprocess(all_recons))['avgpool']
    fake = fake.reshape(len(fake), -1).cpu().numpy()

    swav = np.array([sp.spatial.distance.correlation(gt[i], fake[i]) for i in range(len(gt))]).mean()
    return swav