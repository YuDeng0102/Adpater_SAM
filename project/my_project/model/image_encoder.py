import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from transformers import SamVisionConfig
from transformers.models.sam.modeling_sam import (
    SamVisionEncoderOutput, SamVisionLayer, SamPatchEmbeddings, SamVisionNeck, SamVisionAttention, SamMLPBlock
)
from .My_Modules import Adapter,Linear_Adapter
from .My_Modules import PromptGenerator
from typing import Optional, Tuple, Union

from mmdet.registry import MODELS
class Block(nn.Module):
    def __init__(self, config, window_size,use_block_adapter=True):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = SamVisionAttention(config, window_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SamMLPBlock(config)
        self.window_size = window_size
        self.use_block_adapter =use_block_adapter
        if use_block_adapter:
            self.adapter=Adapter(config.hidden_size)


    def window_partition(self, hidden_states: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Args:
        Partition into non-overlapping windows with padding if needed.
            hidden_states (tensor): input tokens with [batch_size, height, width, channel]. window_size (int): window
            size.

        Returns:
            windows: windows after partition with [batch_size * num_windows, window_size, window_size, channel].
            (pad_height, pad_width): padded height and width before partition
        """
        batch_size, height, width, channel = hidden_states.shape

        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        hidden_states = F.pad(hidden_states, (0, 0, 0, pad_w, 0, pad_h))
        pad_height, pad_width = height + pad_h, width + pad_w

        hidden_states = hidden_states.reshape(
            batch_size, pad_height // window_size, window_size, pad_width // window_size, window_size, channel
        )
        windows = hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, window_size, window_size, channel)
        return windows, (pad_height, pad_width)

    def window_unpartition(
            self, windows: torch.Tensor, window_size: int, padding_shape: Tuple[int, int],
            original_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Args:
        Window unpartition into original sequences and removing padding.
            hidden_states (tensor):
                input tokens with [batch_size * num_windows, window_size, window_size, channel].
            window_size (int):
                window size.
            padding_shape (Tuple):
                padded height and width (pad_height, pad_width).
            original_shape (Tuple): original height and width (height, width) before padding.

        Returns:
            hidden_states: unpartitioned sequences with [batch_size, height, width, channel].
        """
        pad_height, pad_width = padding_shape
        height, width = original_shape
        batch_size = windows.shape[0] // (pad_height * pad_width // window_size // window_size)
        hidden_states = windows.reshape(
            batch_size, pad_height // window_size, pad_width // window_size, window_size, window_size, -1
        )
        hidden_states = (
            hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(batch_size, pad_height, pad_width, -1)
        )

        hidden_states = hidden_states[:, :height, :width, :].contiguous()
        return hidden_states

    @torch.no_grad()
    def layer_norm1_no_grad(self, x):
        return self.layer_norm1(x)

    @torch.no_grad()
    def layer_norm2_no_grad(self, x):
        return self.layer_norm2(x)

    @torch.no_grad()
    def attn_no_grad(self, hidden_states, output_attentions=False):
        return self.attn(hidden_states, output_attentions)

    def forward(
            self,
            hidden_states: torch.Tensor,
    ) -> Tuple[torch.FloatTensor]:

        residual = hidden_states

        hidden_states = self.layer_norm1_no_grad(hidden_states)
        # Window partition
        if self.window_size > 0:
            height, width = hidden_states.shape[1], hidden_states.shape[2]
            hidden_states, padding_shape = self.window_partition(hidden_states, self.window_size)

        hidden_states, attn_weights = self.attn_no_grad(
            hidden_states=hidden_states,
        )

        # Reverse window partition
        if self.window_size > 0:
            hidden_states = self.window_unpartition(hidden_states, self.window_size, padding_shape, (height, width))

        if self.use_block_adapter:
            hidden_states = self.adapter(hidden_states)

        hidden_states = residual + hidden_states

        layernorm_output = self.layer_norm2_no_grad(hidden_states)


        hidden_states = hidden_states + self.mlp(layernorm_output)

        # if self.use_block_adapter:
        #     hidden_states = self.adapter(hidden_states)

        outputs = (hidden_states,)


        return outputs

@MODELS.register_module()
class image_encoder(nn.Module):
    def __init__(self, config: SamVisionConfig):
        super().__init__()
        self.config = config
        self.image_size = config.image_size

        self.patch_embed = SamPatchEmbeddings(config)

        self.pos_embed = None
        if config.use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    config.image_size // config.patch_size,
                    config.image_size // config.patch_size,
                    config.hidden_size,
                )
            )

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            layer= Block(
                config,
                window_size=config.window_size if i not in config.global_attn_indexes else 0,
            )
            self.layers.append(layer)

        self.neck = SamVisionNeck(config)
        self.embed_dim=config.hidden_size

        self.scale_factor = 16
        self.input_type = 'fft'
        self.freq_nums = 0.4

        self.prompt_generator = PromptGenerator(self.scale_factor, self.embed_dim,
                                                config.num_hidden_layers, self.input_type, self.freq_nums,
                                                config.image_size, config.patch_size)
        self.fft_alpha = nn.Parameter(0 * torch.ones((self.embed_dim)), requires_grad=True)


    def get_input_embeddings(self):
        return self.patch_embed

    @torch.no_grad()
    def patch_embed_no_grad(self, x):
        return self.patch_embed(x)

    @torch.enable_grad()
    def patch_embed_grad(self, x):
        return self.patch_embed(x)

    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            patch_embed_grad: Optional[bool] = False,
    ) -> Union[Tuple, SamVisionEncoderOutput]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        inp = pixel_values

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        if patch_embed_grad:
            hidden_states = self.patch_embed_grad(pixel_values)
        else:
            hidden_states = self.patch_embed_no_grad(pixel_values)

        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed

        all_hidden_states = () if output_hidden_states else None


        # fft adapter
        embedding_feature = self.prompt_generator.init_embeddings(hidden_states)
        handcrafted_feature = self.prompt_generator.init_handcrafted(inp)
        prompt = self.prompt_generator.get_prompt(handcrafted_feature, embedding_feature)

        B, H, W, _ = hidden_states.shape

        for i, block in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states=hidden_states+prompt[i].reshape(B,H,W,-1)
            layer_outputs = block(hidden_states)
            hidden_states = layer_outputs[0]


        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.neck(hidden_states)

        if not return_dict:
            outputs = (hidden_states,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            return outputs

        return SamVisionEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )
