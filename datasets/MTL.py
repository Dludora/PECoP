# !/usr/bin/env python
# -*- coding: utf-8 -*-
# 2022/4/2 16:19
import copy
import glob
import os
import pickle
import numpy as np
import random
import cv2
from utils.util import read_pkl
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from PIL import Image


class MTL_Dataset(Dataset):
    """MTL-AQA dataset"""

    def __init__(self, args, transforms_=None, color_jitter_=None):
        super(MTL_Dataset, self).__init__()
        # get_data
        self.data_path = args.data_list
        self.dataset = self.read_pickle(self.data_path)
        # get label
        self.label_dict = read_pkl('MTL-AQA_split_0_data', 'final_annotations_dict.pkl')

        self.rgb_prefix = args.rgb_prefix
        self.clip_len = args.clip_len
        self.max_sr = args.max_sr
        self.max_segment = args.max_segment
        self.fr = args.fr
        self.toPIL = transforms.ToPILImage()
        self.transforms_ = transforms_
        self.color_jitter_ = color_jitter_

    def read_pickle(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            pickle_data = pickle.load(f)

        return pickle_data

    # def load_video(self, video_file_name, phase):
    #     image_list = sorted(
    #         (glob.glob(os.path.join(self.data_root, f'{video_file_name[0]:02d}', '*.jpg')))
    #     )
    #     end_frame = self.label_dict.get(video_file_name).get('end_frame')
    #     if phase == 'train':
    #         temporal_aug_shift = random.randint(self.temporal_shift[0], self.temporal_shift[1])
    #         if end_frame + temporal_aug_shift > self.length or end_frame + temporal_aug_shift < len(image_list):
    #             end_frame = end_frame + temporal_aug_shift
    #     start_frame = end_frame - self.length
    #
    #     video = [Image.open(image_list[start_frame + i]) for i in range(self.length)]
    #     return self.transform_(video)

    def get_start_last_frame(self, video_name):
        end_frame = self.label_dict.get(video_name).get('end_frame')
        start_frame = self.label_dict.get(video_name).get('start_frame')
        frame_num = end_frame - start_frame + 1

        return start_frame, end_frame, frame_num

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample_start_frame, sample_end_frame, sample_frame_num = self.get_start_last_frame(sample)

        sample_rate = random.randint(1, self.max_sr)
        while sample_rate == self.fr:
            sample_rate = random.randint(1, self.max_sr)

        segment = random.randint(1, self.max_segment)
        # 
        clip_start_frame = random.randint(1, sample_frame_num - self.clip_len)

        segment_start_frame = int((segment - 1) * (self.clip_len / self.max_segment))
        segment_last_frame = int(segment * (self.clip_len / self.max_segment))

        rgb_clip = self.load_clip()

        label_speed = sample_rate - 1
        label_segment = segment - 1
        label = [label_speed, label_segment]

        trans_clip = self.transforms_(rgb_clip)

        return trans_clip, np.array(label)

    def load_clip(self, video_dir, start_frame, sample_rate, clip_len,
                  num_frames, segment_start_frame, segment_last_frame):

        video_clip = []
        idx1 = 0
        idx = 0
        normal_f = copy.deepcopy(start_frame)

        for i in len(clip_len):
            if segment_start_frame <= i <= segment_last_frame:
                cur_img_path = os.path.join(
                    video_dir,
                    "img_" + "{:05}.jpg".format(start_frame + idx1 * sample_rate)
                )
                normal_f = (start_frame + (idx1 * sample_rate))
                idx = 1

                img = cv2.imread(cur_img_path)
                video_clip.append(img)

                if (start_frame + (idx1 + 1) * sample_rate) > num_frames:
                    start_frame = 1
                    normal_f = 1
                    idx = 0
                    idx1 = 0
                else:
                    idx1 += 1
            else:
                cur_img_path = os.path.join(
                    video_dir,
                    "img_" + "{:05}.jpg".format(normal_f + idx))

                start_frame = normal_f + idx
                idx1 = 1

                # print(cur_img_path)

                img = cv2.imread(cur_img_path)
                video_clip.append(img)

                if (normal_f + (idx + 4)) > num_frames:
                    normal_f = 1
                    start_frame = 1
                    idx1 = 0
                    idx = 0
                else:
                    idx += self.fr

        video_clip = np.array(video_clip)

        return video_clip

    def __len__(self):
        return len(self.dataset)
