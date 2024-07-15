from models.local.mask2former import Mask2FormerImageProcessorForRefAVS
from models.local.mask2former import Mask2FormerForRefAVS
from models.local.mask2former import logging

from PIL import Image
import requests
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Module
import re
import matplotlib.pyplot as plt

logging.set_verbosity_error()


image_processor = Mask2FormerImageProcessorForRefAVS.from_pretrained("facebook/mask2former-swin-base-ade-semantic")
model_m2f = Mask2FormerForRefAVS.from_pretrained(
    "facebook/mask2former-swin-base-ade-semantic"
)

# avs_dataset = AVS()

class REFAVS_Model_Base(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.model_v = model_m2f.cuda()

        self.dim_v = 1024
        self.num_heads = 8
        
        self.audio_proj = nn.Sequential(
            nn.Linear(128, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.dim_v),
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(768, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.dim_v),
        )

        self.prompt_proj = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 256),
        )

        self.cfgs = cfgs
        
        self.loss_fn = F.binary_cross_entropy_with_logits  # 'bce'

        self.mha_A_T = nn.MultiheadAttention(self.dim_v, self.num_heads)
        self.mha_V_T = nn.MultiheadAttention(self.dim_v, self.num_heads)
        self.mha_mm = nn.MultiheadAttention(self.dim_v, self.num_heads)

        self.cache_mem_beta = 1

    def fusion_mm_to_text(self, feat_a_or_v, feat_t):
        assert feat_a_or_v.shape[-1] == self.dim_v
        assert feat_t.shape[-1] == self.dim_v
        return torch.concat((feat_a_or_v, feat_t), dim=1)

    def process_with_cached_memory(self, feat_mm):
        feat_beta = feat_mm * (self.cache_mem_beta + 1)
        cached_mem = torch.cumsum(feat_mm, dim=0) 
        mean_feat_at_each_time_step = cached_mem / torch.arange(1, feat_mm.shape[0] + 1).view(-1, 1, 1).cuda()
        diff_feat = feat_beta - mean_feat_at_each_time_step
        return diff_feat
    
    def forward(self, batch_data):
        uid, mask_recs, img_recs, image_sizes, feat_aud, feat_text, rec_audio, rec_text = batch_data
        bsz = len(uid)
        frame_n = len(img_recs[0])
        loss_uid = []
        uid_preds = []
        assert len(uid) == len(img_recs) and len(uid) == len(rec_text)

        mask_recs = [torch.stack(rec) for rec in mask_recs]
        gt_label = torch.stack(mask_recs).view(bsz*frame_n, mask_recs[0].shape[-2], mask_recs[0].shape[-1]).squeeze().cuda()

        feat_aud = torch.stack(feat_aud).cuda()
        feat_text = torch.stack(feat_text).cuda()
        feat_aud = self.audio_proj(feat_aud).view(bsz, feat_aud.shape[-2], self.dim_v)
        feat_text = self.text_proj(feat_text).view(bsz, feat_text.shape[-2], self.dim_v)

        batch_pixel_values, batch_pixel_mask = [], []

        for idx, _ in enumerate(uid):
            img_input = img_recs[idx]
            
            for img in img_input:
                batch_pixel_values.append(img['pixel_values'])
                batch_pixel_mask.append(img['pixel_mask'])

        batch_pixel_values = torch.stack(batch_pixel_values).squeeze().cuda()
        batch_pixel_mask = torch.stack(batch_pixel_mask).squeeze().cuda()

        batch_input = {
            'pixel_values': batch_pixel_values,
            'pixel_mask': batch_pixel_mask,
            'mask_labels': gt_label
        }

        outputs = self.model_v(**batch_input)
        feat_vis = outputs['encoder_last_hidden_state'].view(bsz, self.dim_v, 12*12, frame_n).view(bsz, -1, self.dim_v)
     
        fused_T_with_A = self.fusion_mm_to_text(feat_aud, feat_text)   
        fused_T_with_V = self.fusion_mm_to_text(feat_vis, feat_text)   

        fused_T_with_A = fused_T_with_A.permute(1, 0, 2) 
        fused_T_with_V = fused_T_with_V.permute(1, 0, 2)
        
        fused_T_with_A, _ = self.mha_A_T(fused_T_with_A, fused_T_with_A, fused_T_with_A)
        fused_T_with_V, _ = self.mha_V_T(fused_T_with_V, fused_T_with_V, fused_T_with_V)

        fused_T_with_A_part_A, fused_T_with_A_part_T = \
            fused_T_with_A[:feat_aud.shape[1], :, :], fused_T_with_A[feat_aud.shape[1]:, :, :]
        fused_T_with_V_part_V, fused_T_with_V_part_T = \
            fused_T_with_V[:feat_vis.shape[1], :, :], fused_T_with_V[feat_vis.shape[1]:, :, :]

        assert fused_T_with_A_part_A.shape[0] + fused_T_with_A_part_T.shape[0] == fused_T_with_A.shape[0]

        cues_A = self.process_with_cached_memory(fused_T_with_A_part_A).permute(1, 0, 2)  # [bsz, len, dim_v]
        cues_V = self.process_with_cached_memory(fused_T_with_V_part_V).permute(1, 0, 2)
        cues_T = (feat_text + fused_T_with_A_part_T.permute(1, 0, 2) + fused_T_with_V_part_T.permute(1, 0, 2)) \
            / torch.tensor(3.0).cuda()

        tag_A = torch.full([bsz, 1, self.dim_v], 0).cuda()
        tag_V = torch.full([bsz, 1, self.dim_v], 1).cuda()

        cues_V = cues_V.view(bsz, frame_n, 12*12, self.dim_v)

        batch_prompt_emb = []
        for f in range(frame_n):
            cues_V_f = cues_V[:, f]
            cues_mm = torch.concat([cues_A, tag_A, cues_V_f, tag_V, cues_T], dim=1)
            cues_mm, _ = self.mha_mm(cues_mm, cues_mm, cues_mm)
            batch_prompt_emb.append(cues_mm)

        batch_prompt_emb = torch.stack(batch_prompt_emb).permute(1, 0, 2, 3) 
        batch_prompt_emb = batch_prompt_emb.contiguous().view(bsz*frame_n, batch_prompt_emb.shape[-2], self.dim_v)
        batch_prompt_emb = self.prompt_proj(batch_prompt_emb) 

        batch_input = {
            'pixel_values': batch_pixel_values,
            'pixel_mask': batch_pixel_mask,
            'prompt_features_projected': batch_prompt_emb, 
            'mask_labels': gt_label
        }

        outputs = self.model_v(**batch_input)

        pred_instance_map = image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[[256, 256]]*(bsz*frame_n),
        )

        pred_instance_map = torch.stack(pred_instance_map, dim=0).view(bsz*frame_n, 256, 256)

        loss_frame = self.loss_fn(input=pred_instance_map.squeeze(), target=gt_label.squeeze().cuda())
        loss_uid.append(loss_frame)
        uid_preds.append(pred_instance_map.squeeze())

        return loss_uid, uid_preds
