import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, logging
from transformers.models.mask2former.configuration_mask2former import Mask2FormerConfig
from transformers.models.mask2former.modeling_mask2former import Mask2FormerModel, Mask2FormerTransformerModule, Mask2FormerModelOutput, Mask2FormerPixelDecoderEncoderLayer
from transformers.models.mask2former.modeling_mask2former import Mask2FormerPixelLevelModule, Mask2FormerPixelDecoder, Mask2FormerPixelDecoderEncoderOnly
from transformers.models.mask2former.modeling_mask2former import Mask2FormerForUniversalSegmentationOutput, Mask2FormerMaskedAttentionDecoderOutput
from transformers.models.mask2former.modeling_mask2former import Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention
from transformers.models.mask2former.modeling_mask2former import multi_scale_deformable_attention

from typing import Dict, List, Optional, Tuple
from torch import Tensor, nn

from .refavs_transformer import REF_AVS_Transformer


class Mask2FormerTransformerModuleForRefAVS(Mask2FormerTransformerModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        print('>>> Init m2f for refavs...')
        self.ref_avs_attn = REF_AVS_Transformer()
        
    def prefix_tuning(self, prompt, feature):
        feature[:23] = prompt + feature[:2]
        return feature
    
    def check_transformer(self):
        # print('>>> Using new module.')
        ...

        
    def forward(
        self,
        multi_scale_features: List[Tensor],
        mask_features: Tensor,
        prompt_features_projected: Tensor = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> Mask2FormerMaskedAttentionDecoderOutput:

        
        multi_stage_features = []
        multi_stage_positional_embeddings = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(multi_scale_features[i].shape[-2:])
            multi_stage_positional_embeddings.append(self.position_embedder(multi_scale_features[i], None).flatten(2))
            multi_stage_features.append(
                self.input_projections[i](multi_scale_features[i]).flatten(2)
                + self.level_embed.weight[i][None, :, None]
            )

            multi_stage_positional_embeddings[-1] = multi_stage_positional_embeddings[-1].permute(2, 0, 1)
            multi_stage_features[-1] = multi_stage_features[-1].permute(2, 0, 1)

        _, batch_size, _ = multi_stage_features[0].shape

        query_embeddings = self.queries_embedder.weight.unsqueeze(1).repeat(1, batch_size, 1) 
        query_features = self.queries_features.weight.unsqueeze(1).repeat(1, batch_size, 1)
        
        if prompt_features_projected is not None: 
            
            bsz = prompt_features_projected.shape[0]
            num_queries = query_features.shape[0]

            query_features = self.ref_avs_attn(target=query_features, source=prompt_features_projected)

        decoder_output = self.decoder(
            inputs_embeds=query_features,
            multi_stage_positional_embeddings=multi_stage_positional_embeddings,
            pixel_embeddings=mask_features,
            encoder_hidden_states=multi_stage_features,
            query_position_embeddings=query_embeddings,
            feature_size_list=size_list,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

        return decoder_output

class Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttentionForRefAVS(Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention):
    def __init__(self, embed_dim: int, num_heads: int, n_levels: int, n_points: int):
        super().__init__(embed_dim, num_heads, n_levels, n_points)
        self.avs_adapt = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//4),
            nn.ReLU(),
            nn.Linear(embed_dim//4, embed_dim),
        )
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        output_attentions: bool = False,
    ):
        # add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None:
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        if (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() != sequence_length:
            raise ValueError(
                "Make sure to align the spatial shapes with the sequence length of the encoder hidden states"
            )

        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            # we invert the attention_mask
            value = value.masked_fill(attention_mask[..., None], float(0))
        value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = nn.functional.softmax(attention_weights, -1).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points
        )
        # batch_size, num_queries, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")

        output = multi_scale_deformable_attention(value, spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        
        output = self.avs_adapt(output)

        return output, attention_weights


class Mask2FormerPixelDecoderEncoderLayerForRefAVS(Mask2FormerPixelDecoderEncoderLayer):
    def __init__(self, config: Mask2FormerConfig):
        super().__init__(config)
        self.self_attn = Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttentionForRefAVS(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            n_levels=3,
            n_points=4,
        )
    
class Mask2FormerPixelDecoderEncoderOnlyForRefAVS(Mask2FormerPixelDecoderEncoderOnly):
    def __init__(self, config: Mask2FormerConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
                [Mask2FormerPixelDecoderEncoderLayerForRefAVS(config) for _ in range(config.encoder_layers)]
            )

class Mask2FormerPixelDecoderForRefAVS(Mask2FormerPixelDecoder):
    def __init__(self, config: Mask2FormerConfig, feature_channels):
        super().__init__(config, feature_channels)
        self.encoder = Mask2FormerPixelDecoderEncoderOnlyForRefAVS(config)

class Mask2FormerPixelLevelModuleForRefAVS(Mask2FormerPixelLevelModule):
    def __init__(self, config: Mask2FormerConfig):
        super().__init__(config)
        self.decoder = Mask2FormerPixelDecoderForRefAVS(config, feature_channels=self.encoder.channels)


class Mask2FormerModelForRefAVS(Mask2FormerModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        self.pixel_level_module = Mask2FormerPixelLevelModuleForRefAVS(config)
        self.transformer_module = Mask2FormerTransformerModuleForRefAVS(in_features=config.feature_size, config=config) 
        
    def forward(
        self,
        pixel_values: Tensor,
        pixel_mask: Optional[Tensor] = None,
        prompt_features_projected: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # memory_last_hidden=None,
    ) -> Mask2FormerModelOutput:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, _, height, width = pixel_values.shape

        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), device=pixel_values.device)

        pixel_level_module_output = self.pixel_level_module(
            pixel_values=pixel_values, output_hidden_states=output_hidden_states,
        ) 


        transformer_module_output = self.transformer_module(
            prompt_features_projected=prompt_features_projected,
            multi_scale_features=pixel_level_module_output.decoder_hidden_states,
            mask_features=pixel_level_module_output.decoder_last_hidden_state,
            output_hidden_states=True,
            output_attentions=output_attentions,
        )

        encoder_hidden_states = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None
        transformer_decoder_intermediate_states = None

        if output_hidden_states:
            encoder_hidden_states = pixel_level_module_output.encoder_hidden_states
            pixel_decoder_hidden_states = pixel_level_module_output.decoder_hidden_states
            transformer_decoder_hidden_states = transformer_module_output.hidden_states
            transformer_decoder_intermediate_states = transformer_module_output.intermediate_hidden_states

        output = Mask2FormerModelOutput(
            encoder_last_hidden_state=pixel_level_module_output.encoder_last_hidden_state,
            pixel_decoder_last_hidden_state=pixel_level_module_output.decoder_last_hidden_state,
            transformer_decoder_last_hidden_state=transformer_module_output.last_hidden_state,
            encoder_hidden_states=encoder_hidden_states,
            pixel_decoder_hidden_states=pixel_decoder_hidden_states,
            transformer_decoder_hidden_states=transformer_decoder_hidden_states,
            transformer_decoder_intermediate_states=transformer_decoder_intermediate_states,
            attentions=transformer_module_output.attentions,
            masks_queries_logits=transformer_module_output.masks_queries_logits,
        )

        if not return_dict:
            output = tuple(v for v in output.values() if v is not None)

        return output


         
class Mask2FormerForRefAVS(Mask2FormerForUniversalSegmentation):
    def __init__(self, config: Mask2FormerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.model = Mask2FormerModelForRefAVS(config)
    
    def forward(
        self,
        pixel_values: Tensor,
        prompt_features_projected: Optional[Tensor] = None,
        mask_labels: Optional[List[Tensor]] = None,
        class_labels: Optional[List[Tensor]] = None,
        pixel_mask: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_auxiliary_logits: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # memory_last_hidden=None,
    ) -> Mask2FormerForUniversalSegmentationOutput:
 
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            prompt_features_projected=prompt_features_projected,
            output_hidden_states=output_hidden_states or self.config.use_auxiliary_loss,
            output_attentions=output_attentions,
            return_dict=True,
        )

        loss, loss_dict, auxiliary_logits = None, None, None
        class_queries_logits = ()

        for decoder_output in outputs.transformer_decoder_intermediate_states:
            class_prediction = self.class_predictor(decoder_output.transpose(0, 1))
            class_queries_logits += (class_prediction,)

        masks_queries_logits = outputs.masks_queries_logits

        auxiliary_logits = self.get_auxiliary_logits(class_queries_logits, masks_queries_logits)

        if mask_labels is not None and class_labels is not None:
            loss_dict = self.get_loss_dict(
                masks_queries_logits=masks_queries_logits[-1],
                class_queries_logits=class_queries_logits[-1],
                mask_labels=mask_labels,
                class_labels=class_labels,
                auxiliary_predictions=auxiliary_logits,
            )
            loss = self.get_loss(loss_dict)

        encoder_hidden_states = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None

        if output_hidden_states:
            encoder_hidden_states = outputs.encoder_hidden_states
            pixel_decoder_hidden_states = outputs.pixel_decoder_hidden_states
            transformer_decoder_hidden_states = outputs.transformer_decoder_hidden_states

        output_auxiliary_logits = (
            self.config.output_auxiliary_logits if output_auxiliary_logits is None else output_auxiliary_logits
        )
        if not output_auxiliary_logits:
            auxiliary_logits = None

        output = Mask2FormerForUniversalSegmentationOutput(
            loss=loss,
            class_queries_logits=class_queries_logits[-1],
            masks_queries_logits=masks_queries_logits[-1],
            auxiliary_logits=auxiliary_logits,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            pixel_decoder_last_hidden_state=outputs.pixel_decoder_last_hidden_state,
            transformer_decoder_last_hidden_state=outputs.transformer_decoder_last_hidden_state,
            encoder_hidden_states=encoder_hidden_states,
            pixel_decoder_hidden_states=pixel_decoder_hidden_states,
            transformer_decoder_hidden_states=transformer_decoder_hidden_states,
            attentions=outputs.attentions,
        )

        if not return_dict:
            output = tuple(v for v in output.values() if v is not None)
            if loss is not None:
                output = ((loss)) + output
        return output
        
