import torch
from transformers.models.mask2former.image_processing_mask2former import Mask2FormerImageProcessor

from typing import Dict, List, Optional, Tuple
from torch import Tensor, nn

class Mask2FormerImageProcessorForRefAVS(Mask2FormerImageProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def post_process_semantic_segmentation(
        self, outputs, target_sizes: Optional[List[Tuple[int, int]]] = None
    ) -> "torch.Tensor":
        """
        Converts the output of [`Mask2FormerForUniversalSegmentation`] into semantic segmentation maps. Only supports
        PyTorch.

        Args:
            outputs ([`Mask2FormerForUniversalSegmentation`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple[int, int]]`, *optional*):
                List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
                final size (height, width) of each prediction. If left to None, predictions will not be resized.
        Returns:
            `List[torch.Tensor]`:
                A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width)
                corresponding to the target_sizes entry (if `target_sizes` is specified). Each entry of each
                `torch.Tensor` correspond to a semantic class id.
        """
        AVS_BINARY = True
        
        class_queries_logits = outputs.class_queries_logits
        bsz = class_queries_logits.shape[0]
        if AVS_BINARY:
            class_queries_logits = outputs.class_queries_logits[:, :, 0].view(bsz, 100, 1)
            null_queries_logits = outputs.class_queries_logits[:, :, -1].view(bsz, 100, 1)
            class_queries_logits = torch.concat([class_queries_logits, null_queries_logits], dim=-1)
  
        masks_queries_logits = outputs.masks_queries_logits 

        # Scale back to preprocessed image size - (384, 384) for all models
        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False
        )

        # Remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]  # [1, 100, 1]

        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]
        if AVS_BINARY:
            masks_probs = masks_queries_logits  # .sigmoid()
        
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        batch_size = class_queries_logits.shape[0]

        # Resize logits and compute semantic segmentation maps
        if target_sizes is not None:
            if batch_size != len(target_sizes):
                print(f'bsz: {batch_size} | target: {target_sizes}')
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            semantic_segmentation = []
            for idx in range(batch_size):
                resized_logits = torch.nn.functional.interpolate(
                    segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                      
                semantic_map = resized_logits[0].argmax(dim=0)
                if AVS_BINARY:
                    semantic_map = resized_logits[0]
     
                semantic_segmentation.append(semantic_map)
        else:
            
            semantic_segmentation = segmentation.argmax(dim=1)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        return semantic_segmentation