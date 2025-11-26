import os, random
import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import open_clip
from pycocotools.coco import COCO
import pandas as pd
from pandas import json_normalize



def normalize_to_neg_one_to_one(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = 2 * ((tensor - min_val) / (max_val - min_val)) - 1.0
    return normalized_tensor

class NSD_ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir_nsd='../Datasets/NSD_fsLR',
                 subjs=[1],
                 subjs_nsd=[1],
                 image_norm=True,
                 phase='train', val_data_fraction=1.0):

        test_image_path = f'{root_dir_nsd}/nsddata_stimuli/stimuli/images'
        annotation_file = f'{root_dir_nsd}/nsddata_stimuli/instances_train2017.json'
        caption_file = f'{root_dir_nsd}/nsddata_stimuli/captions_train2017.json'
        annotation_file_val = f'{root_dir_nsd}/nsddata_stimuli/instances_val2017.json'
        caption_file_val = f'{root_dir_nsd}/nsddata_stimuli/captions_val2017.json'
        nsd_info_file = f'{root_dir_nsd}/nsddata_stimuli/nsd_stim_info_merged.csv'
        data_path_nsd = f'{root_dir_nsd}/fmri_npy'
        test_cap_label_path = f'{root_dir_nsd}/COCO_73k_annots_curated.npy'
        shared_trial = f'{root_dir_nsd}/sub1257_shared_triallabel.npy'
        unique_trial = f'{root_dir_nsd}/unique_triallabel.npy'

        self.coco_train = COCO(annotation_file)
        self.coco_val = COCO(annotation_file_val)

        self.coco_cap_train = COCO(caption_file)
        self.coco_cap_val = COCO(caption_file_val)

        self.stim_descriptions = pd.read_csv(nsd_info_file, index_col=0)

        self.cat_ids = self.coco_train.getCatIds()
        self.num_classes = len(self.cat_ids)

        print(f"\033[92m COCO num_classes: {self.num_classes} \033[0m")


        image_transform_list = [transforms.Resize((224, 224))]
        image_transform_list.append(transforms.ToTensor())
        # if image_norm:
        #     image_transform_list.append(transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)))
        self.image_transform = transforms.Compose(image_transform_list)
        self.fmri_transform = transforms.Compose([transforms.Normalize(mean=0.5, std=0.5)])

        self.image_norm = image_norm
        self.data_path_nsd = Path(data_path_nsd)
        self.data = dict()
        self.images_all_subjs = dict()
        self.image_subj_fmri_dict = dict()
        self.unique_images = []
        self.unique_trial_dict = dict()
        self.shared_trial_dict = dict()
        self.test_images_path = test_image_path
        self.test_cap_label = np.load(test_cap_label_path, allow_pickle=True)  # caption file
        self.sub = subjs
        self.sub_nsd = subjs_nsd
        self.phase = phase

        self.clip_tokenizer = open_clip.tokenize
        self.max_seq_len = 30


        if phase == 'train':
            # ======================================================
            # Unique Trial
            # ======================================================
            train_label = np.load(unique_trial, allow_pickle=True)  # train index for each subject in dict
            self.train_dict_map = dict()
            temp_image_list = []
            last = 0
            for subj in self.sub_nsd:
                temp_image = np.array(list(train_label.item()[subj - 1]))  # start from 0, subj index -1
                temp_image_list.append(temp_image)
                temp_data = np.array(np.load(self.data_path_nsd / f'{subj:02d}_label.npy'))  # fmri -> image idx
                label_dict = {}  # reverse temp_data: image idx -> fmri
                for fmri_idx in range(len(temp_data)):
                    label_dict[temp_data[fmri_idx]] = fmri_idx
                self.unique_trial_dict[subj] = label_dict

                for img_idx in range(len(temp_image)):
                    self.train_dict_map[img_idx + last] = subj
                last += len(temp_image)

            self.unique_images = np.concatenate(temp_image_list)




        else:
            for subj_idx in self.sub:
                images_cur_subj = np.array(np.load(self.data_path_nsd / f'{subj_idx:02d}_label.npy'))
                # print(f"\033[92m {images_cur_subj} \033[0m")
                self.images_all_subjs[subj_idx] = images_cur_subj

            val_image_idx = np.load(shared_trial, allow_pickle=True)  # val label for one subject
            val_image_idx = np.array(val_image_idx)

            # print(f"\033[92m {len(val_image_idx)} \033[0m")
            # validate using part of the val dataset
            val_image_num = int(val_data_fraction * len(val_image_idx))
            val_image_idx = val_image_idx[:val_image_num]

            # print(f"\033[95m val_image {len(val_image_idx)} \033[0m")

            temp = []
            last = 0
            for subj_idx in self.sub:
                val_label = []
                for idx in val_image_idx:
                    where_result = np.where(self.images_all_subjs[subj_idx] == idx)[0]
                    if len(where_result) > 0:  # Check if the condition is met
                        val_label.append(idx)
                temp.append(val_label)
                last += len(val_label)

                image_to_fmri_idx_dict = {}
                for fmri_idx in range(len(self.images_all_subjs[subj_idx])):
                    image_to_fmri_idx_dict[self.images_all_subjs[subj_idx][fmri_idx]] = fmri_idx

                # print(f"\033[92m {len(image_to_fmri_idx_dict)} \033[0m")
                self.shared_trial_dict[subj_idx] = image_to_fmri_idx_dict
            self.val_label = np.concatenate(temp)


        if phase == 'pretrain' and phase == 'train':
            self.is_train = True
        else:
            self.is_train = False
        print(f'Data length:{self.__len__()}')

    def pad_data(self, images, max_length):
        for subj in images:
            current_length = len(images[subj])

            # print(f"\033[96m {current_length} \033[0m")

            if current_length < max_length:
                padding_length = max_length - current_length

                original_imgs = images[subj]
                repeat_times = (padding_length // current_length) + 1
                padding_imgs = np.tile(original_imgs, repeat_times)[:padding_length]
                images[subj] = np.concatenate([original_imgs, padding_imgs])

        return images


    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64)))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        return tokens


    def __getitem__(self, i):

        if self.phase == "train":
            # =====================================================================================
            # Unique trials
            # =====================================================================================

            idx = int(self.unique_images[i])
            sub_idx = self.train_dict_map[i]
            fMRI_idx = self.unique_trial_dict[sub_idx][idx]
            fMRI = np.load(self.data_path_nsd / f'{sub_idx:02d}_norm/surf_{fMRI_idx:06d}.npy')
            fMRI = torch.from_numpy(fMRI)[None]

            image_filename_u = os.path.join(self.test_images_path, f'image_{idx:06d}.png')
            natural_image_u = Image.open(image_filename_u)
            gt_image_u = self.image_transform(natural_image_u)

            annots_u = self.test_cap_label[idx]
            caption_u = list(annots_u[annots_u != ''])
            random_caption_u = random.choice(caption_u)

            gpt_token_unique = self.clip_tokenizer(random_caption_u)[0].clone().to(dtype=torch.int64)


            ### load coco labels
            subj_info = self.stim_descriptions.iloc[idx]
            coco_id = subj_info['cocoId']
            labels = torch.zeros(self.num_classes, dtype=torch.float32)
            if "train" in subj_info['cocoSplit']:
                cat_ids = self.coco_train.getCatIds()
                categories = json_normalize(self.coco_train.loadCats(cat_ids))
                coco_cats = []
                coco_cat_ids = []
                coco_cat_idx = []
                max_area = 0
                best_cat_idx = -1

                for cat_id in cat_ids:
                    this_img_list = self.coco_train.getImgIds(catIds=[cat_id])
                    if coco_id in this_img_list:
                        this_cat = np.asarray(categories[categories['id'] == cat_id]['name'])[0]
                        cat_idx = cat_ids.index(cat_id)

                        ann_ids = self.coco_train.getAnnIds(imgIds=coco_id, catIds=[cat_id])
                        annotations = self.coco_train.loadAnns(ann_ids)

                        for ann in annotations:
                            if ann['area'] > max_area:
                                max_area = ann['area']
                                best_cat_idx = cat_idx
                                best_cat = this_cat

                        coco_cats.append(this_cat)
                        coco_cat_ids.append(cat_id)
                        coco_cat_idx.append(cat_idx)
                        labels[cat_idx] = 1.0

                if best_cat_idx != -1:
                    labels[:] = 0
                    labels[best_cat_idx] = 1.0

            else:
                cat_ids = self.coco_val.getCatIds()
                categories = json_normalize(self.coco_val.loadCats(cat_ids))
                coco_cats = []
                coco_cat_ids = []
                coco_cat_idx = []
                max_area = 0
                best_cat_idx = -1

                for cat_id in cat_ids:
                    this_img_list = self.coco_val.getImgIds(catIds=[cat_id])
                    if coco_id in this_img_list:
                        this_cat = np.asarray(categories[categories['id'] == cat_id]['name'])[0]
                        cat_idx = cat_ids.index(cat_id)

                        ann_ids = self.coco_val.getAnnIds(imgIds=coco_id, catIds=[cat_id])
                        annotations = self.coco_val.loadAnns(ann_ids)

                        for ann in annotations:
                            if ann['area'] > max_area:
                                max_area = ann['area']
                                best_cat_idx = cat_idx
                                best_cat = this_cat


                        coco_cats.append(this_cat)
                        coco_cat_ids.append(cat_id)
                        coco_cat_idx.append(cat_idx)
                        labels[cat_idx] = 1.0


                if best_cat_idx != -1:
                    labels[:] = 0
                    labels[best_cat_idx] = 1.0

            return {
                "fMRIs": fMRI, "subj_lbl": sub_idx-1, "txt": random_caption_u,
                "gpt_token":gpt_token_unique, "gt_image": gt_image_u, "cls_labels": best_cat_idx, "multi_cls_labels": labels
            }


        else:
            idx = int(self.val_label[i])
            sub_idx = self.sub[0]
            fMRI_idx = self.shared_trial_dict[sub_idx][idx]  # fmri index

            fMRI = np.load(self.data_path_nsd / f'{self.sub[0]:02d}_norm/surf_{fMRI_idx:06d}.npy')

            fMRIs = fMRI[None]
            fMRIs = torch.from_numpy(fMRIs)[None]
            # print(f"\033[91m {surface.shape} \033[0m")

            # gt images
            image_filename = os.path.join(self.test_images_path, f'image_{idx:06d}.png')
            natural_image = Image.open(image_filename).convert('RGB')
            gt_image = self.image_transform(natural_image)

            # coco caption
            annots = self.test_cap_label[idx]
            caption = list(annots[annots != ''])
            # print(f"\033[91m {caption} \033[0m")
            random_caption = random.choice(caption)
            gpt_token = self.clip_tokenizer(random_caption)[0].clone().to(dtype=torch.int64)

            ### load coco labels
            subj_info = self.stim_descriptions.iloc[idx]
            coco_id = subj_info['cocoId']
            labels = torch.zeros(self.num_classes, dtype=torch.float32)
            if "train" in subj_info['cocoSplit']:
                cat_ids = self.coco_train.getCatIds()
                categories = json_normalize(self.coco_train.loadCats(cat_ids))
                coco_cats = []
                coco_cat_ids = []
                max_area = 0
                best_cat_idx = -1

                for cat_id in cat_ids:
                    this_img_list = self.coco_train.getImgIds(catIds=[cat_id])
                    if coco_id in this_img_list:
                        this_cat = np.asarray(categories[categories['id'] == cat_id]['name'])[0]
                        cat_idx = cat_ids.index(cat_id)

                        ann_ids = self.coco_train.getAnnIds(imgIds=coco_id, catIds=[cat_id])
                        annotations = self.coco_train.loadAnns(ann_ids)

                        for ann in annotations:
                            if ann['area'] > max_area:
                                max_area = ann['area']
                                best_cat_idx = cat_idx

                        coco_cats.append(this_cat)
                        coco_cat_ids.append(cat_id)
                        labels[cat_idx] = 1.0
                if best_cat_idx != -1:
                    labels[:] = 0
                    labels[best_cat_idx] = 1.0

            else:
                cat_ids = self.coco_val.getCatIds()
                categories = json_normalize(self.coco_val.loadCats(cat_ids))
                coco_cats = []
                coco_cat_ids = []
                max_area = 0
                best_cat_idx = -1
                for cat_id in cat_ids:
                    this_img_list = self.coco_val.getImgIds(catIds=[cat_id])
                    if coco_id in this_img_list:
                        this_cat = np.asarray(categories[categories['id'] == cat_id]['name'])[0]
                        cat_idx = cat_ids.index(cat_id)

                        ann_ids = self.coco_val.getAnnIds(imgIds=coco_id, catIds=[cat_id])
                        annotations = self.coco_val.loadAnns(ann_ids)

                        for ann in annotations:
                            if ann['area'] > max_area:
                                max_area = ann['area']
                                best_cat_idx = cat_idx

                        coco_cats.append(this_cat)
                        coco_cat_ids.append(cat_id)
                        labels[cat_idx] = 1.0

                if best_cat_idx != -1:
                    labels[:] = 0
                    labels[best_cat_idx] = 1.0

            return {
                "fMRIs": fMRIs, "txt": random_caption, "gpt_token": gpt_token,
                "gt_image": gt_image, "cls_labels": best_cat_idx, "multi_cls_labels": labels
            }

    def __len__(self):
        if self.phase == "train":
            return len(self.unique_images)
        else:
            return len(self.val_label)

