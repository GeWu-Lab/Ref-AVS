import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pdb

import sys
import os
import random

from torchvision import transforms
from collections import defaultdict
import cv2
from transformers import AutoImageProcessor, AutoTokenizer, AutoModel
from PIL import Image

from towhee import pipe, ops
from transformers import pipeline


# logger = log_agent('audio_recs.log')

import pickle as pkl

class REFAVS(Dataset):
    def __init__(self, split='train', cfg=None):
        # metadata: train/test/val
        self.data_dir = cfg.data_dir
        meta_path = f'{self.data_dir}/metadata.csv'
        metadata = pd.read_csv(meta_path, header=0)
        self.split = split
        self.metadata = metadata[metadata['split'] == split]  # split= train,test,val.

        self.media_path = f'{self.data_dir}/media'
        self.label_path = f'{self.data_dir}/gt_mask'
        self.frame_num = cfg.frame_n
        self.text_max_len = cfg.text_max_len

        # modalities processor/pipelines
        self.img_process = AutoImageProcessor.from_pretrained(cfg.m2f_model)

        self.audio_vggish_pipeline = (   # pipeline building
            pipe.input('path')
                .map('path', 'frame', ops.audio_decode.ffmpeg())
                .map('frame', 'vecs', ops.audio_embedding.vggish())
                .output('vecs')
        )

        self.text_tokenizer = AutoTokenizer.from_pretrained(cfg.text_model)
        self.text_encoder = AutoModel.from_pretrained(cfg.text_model).cuda().eval()

    def get_audio_emb(self, wav_path):
        """ wav string path. """ 
        emb = torch.tensor(self.audio_vggish_pipeline(wav_path).get()[0])
        # print(len(emb))
        return emb
    
    def get_text_emb(self, exp):
        """ readable textual reference. """
        inputs = self.text_tokenizer(exp, max_length=25, padding="max_length", truncation=True, return_tensors="pt")
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        with torch.no_grad():
            emb = self.text_encoder(**inputs).last_hidden_state  # [1, max_len, 768]
        return emb

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        df_one_video = self.metadata.iloc[idx]
        vid, uid, fid, exp = df_one_video['vid'], df_one_video['uid'], df_one_video['fid'], df_one_video['exp']  # uid for vid.
        vid = uid.rsplit('_', 2)[0]  # TODO: use encoded id.

        img_recs = []
        mask_recs = []
        images = []

        rec_audio = f'{self.media_path}/{vid}/audio.wav'
        rec_text = exp

        feat_aud = self.get_audio_emb(rec_audio)
        feat_text = self.get_text_emb(rec_text)

        for _idx in range(self.frame_num):  # set frame_num as the batch_size
            # frame 
            path_frame = f'{self.media_path}/{vid}/frames/{_idx}.jpg'  # image
            image = Image.open(path_frame)
            image_sizes = [image.size[::-1]]
            image_inputs = self.img_process(image, return_tensors="pt")  # singe frame rec
            
            # mask label
            path_mask = f'{self.label_path}/{vid}/fid_{fid}/0000{_idx}.png'  # new
            mask_cv2 = cv2.imread(path_mask)
            mask_cv2 = cv2.resize(mask_cv2, (256, 256))
            mask_cv2 = cv2.cvtColor(mask_cv2, cv2.COLOR_BGR2GRAY)
            gt_binary_mask = torch.as_tensor(mask_cv2 > 0, dtype=torch.float32)
            
            # video frames collect
            img_recs.append(image_inputs)
            mask_recs.append(gt_binary_mask)
        
        return vid, mask_recs, img_recs, image_sizes, feat_aud, feat_text, rec_audio, rec_text