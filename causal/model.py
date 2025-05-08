# author：wenzhuo
# date: 2021 Jun 27 Monday
# based on https://github.com/rn5l/session-rec/blob/master/algorithms/knn/uvsknn.py commit 9f719a6 21/05/26
from collections import defaultdict
from operator import itemgetter
from tqdm import tqdm
import time
import copy
import pandas as pd
from math import log10
import torch.nn as nn

# 新增对比损失类
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negative):
        """
        参数说明：
        - anchor: [B, L-1, D]
        - positive: [B, L-1, D]
        - negative: [B, L-1, N, D]
        """
        # --- 正样本相似度 ---
        pos_sim = F.cosine_similarity(anchor, positive, dim=-1)  # [B, L-1]

        # --- 负样本相似度 ---
        anchor_expanded = anchor.unsqueeze(2)  # [B, L-1, 1, D]
        neg_sim = F.cosine_similarity(anchor_expanded, negative, dim=-1)  # [B, L-1, N]

        # --- 对比损失计算 ---
        logits = torch.cat([
            pos_sim.unsqueeze(-1),  # [B, L-1, 1]
            neg_sim  # [B, L-1, N]
        ], dim=-1) / self.temperature  # [B, L-1, N+1]

        labels = torch.zeros((anchor.size(0), anchor.size(1)), dtype=torch.long, device=anchor.device)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss


# Mixer BEGIN

from typing import Union, Optional
import math
import random
from functools import partial
from torch import Tensor
import numpy as np
import torch
import torch.nn as nn
from timm.layers import DropPath


try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim) # 给Mamba函数传递一个参数：模型维度; Mamba的__init__中,只有模型维度d_model没有默认值,所以需要传递一个参数
        self.norm = norm_cls(dim)

        # drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (self.drop_path(hidden_states) + residual) if residual is not None else hidden_states  # 如果residual是None的话, 就将hidden_states的值赋予residual; 如果residual不是None的话,就将hidden_states与residual相加,作为新的residual
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype)) # 对residual执行正则化, 输出保存为hidden_states
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params) # Mamba函数的执行,得到输出hidden_states:(B,L,D)
        return hidden_states, residual # 返回隐藏状态和残差, 没有在输出的时候相加, 而是在函数刚开始执行的时候进行相加的

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    # 创建 mamba, “Ctrl+鼠标左键” 查看ssm库中的mamba函数,如果想看部分代码的解释,可以在我们的mamba_ssm_self中寻找到对应的mamba_simple中的mamba函数
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )

    # 初始化Mamba block,包含mamba/Norm/residual Connect
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block


def shuffle_forward(x, residual, layer: nn.Module, inference_params=None, prob: float = 0.0, training: bool = False):
    """
    Forward pass with optional shuffling of the sequence dimension.

    Args:
    - x (torch.Tensor): Input tensor with shape (B, L, d).
    - residual: Input tensor of the same size of x, required by mamba model
    - layer (nn.Module): A PyTorch module through which x should be passed.
    - prob (float): Probability of shuffling the sequence dimension L.
    - training (bool): Indicates whether the model is in training mode.

    Returns:
    - torch.Tensor: Output tensor from layer, with the sequence dimension
                    potentially shuffled and then restored.
    """

    B, L, _ = x.shape
    if training and torch.rand(1).item() < prob:
        # Generate a random permutation of indices
        shuffled_indices = torch.randperm(L, device=x.device).repeat(B, 1) # randperm: 生成一个从0到L的随机排列的整数序列;  复制B次, 使Batch中的每个样本都按照这个顺序: (L)-repeat->(B,L)
        # Get inverse indices by sorting the shuffled indices
        inverse_indices = torch.argsort(shuffled_indices, dim=1) # 对打乱后的索引进行排序,从而生成逆序索引,用于恢复打乱前的顺序。

        # Apply the permutation to shuffle the sequence
        x_permuted = x.gather(1, shuffled_indices.unsqueeze(-1).expand(-1, -1, x.size(2))) # 1)沿着x的L方向,根据shuffled_indices打乱顺序;   2)(B,L)-unsqueeze->(B,L,1)-expand->(B,L,C); 对于每个通道来说,都是一样的顺序
        if residual is not None:
            residual_permuted = residual.gather(1, shuffled_indices.unsqueeze(-1).expand(-1, -1, x.size(2))) # 如果residual存在,则也使用相同的打乱索引对residual 进行打乱
        else:
            residual_permuted = residual

         # Forward pass through the layer
        output_permuted, residual_permuted = layer(x_permuted, residual_permuted, inference_params=inference_params) # x_permuted:(B,L,C)
        # Restore the original order
        output = output_permuted.gather(1, inverse_indices.unsqueeze(-1).expand(-1, -1, output_permuted.size(2))) # 恢复原始顺序
        residual = residual_permuted.gather(1, inverse_indices.unsqueeze(-1).expand(-1, -1, residual_permuted.size(2))) # 恢复原始顺序
    else:
        # Forward pass without shuffling
        output, residual = layer(x, residual, inference_params=inference_params)

    return output, residual


class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out_in_block: int = 0.,
            drop_path: int = 0.1,
            device=None,
            dtype=None,
            training=True,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        # 创建N个包含Mamba的block,每个block包含(Mamba-norm-drop);  调用create_block函数返回的是定义的Block函数,因此调用self.layer(x)的时候,相当于在执行Block(x)
        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_out_in_block = nn.Dropout(drop_out_in_block) if drop_out_in_block > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, pos=None, inference_params=None):
        hidden_states = input_ids  # + pos
        residual = None
        if pos is None:
            hidden_states = hidden_states
        else:
            hidden_states = hidden_states + pos
        for layer in self.layers:
            # hidden_states, residual = layer(
            #     hidden_states, residual, inference_params=inference_params
            # ) # 循环执行每一个Block;  输出hidden_states:(B,L,D)  输出residual:(B,L,D)

            # 使序列乱序, 然后执行Mamba, 并恢复正确顺序
            hidden_states, residual = shuffle_forward(hidden_states, residual, layer,
                                                      inference_params=inference_params,
                                                      prob=1, training=True)

            hidden_states = self.drop_out_in_block(hidden_states)
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states # 为输出添加残差连接: (B,L,D)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype)) # 为输出添加正则化 (B,L,D)
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states

# Mixer END


# ConBi BEGIN

import torch
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torch import Tensor
import torch.nn.init as init
from mamba_ssm.modules.mamba_simple import Mamba

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)


class PointwiseConv1d(nn.Module):
    """
    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector

    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by pointwise 1-D convolution.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class DepthwiseConv1d(nn.Module):
    """
    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector

    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by depthwise 1-D convolution.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class Swish(nn.Module):
    """
    Swish is a smooth, non-monotonic function that consistently matches or outperforms ReLU on deep networks applied
    to a variety of challenging domains such as Image classification and Machine translation.
    """
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()


class GLU(nn.Module):
    """
    The gating mechanism is called Gated Linear Units (GLU), which was first introduced for natural language processing
    in the paper “Language Modeling with Gated Convolutional Networks”
    """
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()


class Linear(nn.Module):
    """
    Wrapper class of torch.nn.Linear
    Weight initialize by xavier initialization and bias initialize to zeros.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class ResidualConnectionModule(nn.Module):
    """
    Residual Connection Module.
    outputs = (module(inputs) x module_factor + inputs x input_factor)
    """
    def __init__(self, module: nn.Module, module_factor: float = 1.0, input_factor: float = 1.0):
        super(ResidualConnectionModule, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, inputs: Tensor) -> Tensor:
        # 执行传递的模块[FeedForwardModule/ExBimamba/ConformerConvModule], 并添加残差连接,这里的 x_factor可以看作权重
        return (self.module(inputs) * self.module_factor) + (inputs * self.input_factor)


class FeedForwardModule(nn.Module):
    """
    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor contains input sequences

    Outputs: outputs
        - **outputs** (batch, time, dim): Tensor produces by feed forward module.
    """
    def __init__(
            self,
            encoder_dim: int = 512,
            expansion_factor: int = 4,
            dropout_p: float = 0.1,
    ) -> None:
        super(FeedForwardModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            Swish(),
            nn.Dropout(p=dropout_p),
            Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        out = self.sequential(inputs)
        return out


class ConformerConvModule(nn.Module):
    """
    Inputs: inputs
        inputs (batch, time, dim): Tensor contains input sequences

    Outputs: outputs
        outputs (batch, time, dim): Tensor produces by conformer convolution module.
    """
    def __init__(
            self,
            in_channels: int,
            kernel_size: int = 31,
            expansion_factor: int = 2,
            dropout_p: float = 0.1,
    ) -> None:
        super(ConformerConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.sequential = nn.Sequential(
            nn.LayerNorm(in_channels),
            Transpose(shape=(1, 2)),
            PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True),
            GLU(dim=1),
            DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(in_channels),
            Swish(),
            PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        out = self.sequential(inputs).transpose(1, 2)
        return out


class ExBimamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            device=None,
            dtype=None,
            Amatrix_type='default'
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.forward_mamba = Mamba(d_model=self.d_model, d_state=self.d_state, d_conv=self.d_conv, expand=self.expand)
        self.backward_mamba = Mamba(d_model=self.d_model, d_state=self.d_state, d_conv=self.d_conv, expand=self.expand)
        self.output_proj = nn.Linear(2 * self.d_model, self.d_model)

    def forward(self, hidden_input):
        forward_output = self.forward_mamba(hidden_input) # 执行Mamba: (B,T,C)-->(B,T,C)
        backward_output = self.backward_mamba(hidden_input.flip([1])) # 将序列翻转,执行Mamba: (B,T,C)-->(B,T,C)
        res = torch.cat((forward_output, backward_output.flip([1])), dim=-1) # 将序列重新翻转为正的,然后拼接: (B,T,2C)
        res = self.output_proj(res) # 恢复与输入相同的shape:(B,T,2C)-->(B,T,C)
        return res


class ConbimambaBlock(nn.Module):
    """
    Conformer block contains two Feed Forward modules sandwiching the Multi-Headed Self-Attention module
    and the Convolution module. This sandwich structure is inspired by Macaron-Net, which proposes replacing
    the original feed-forward layer in the Transformer block into two half-step feed-forward layers,
    one before the attention layer and one after.

    Args:
        encoder_dim (int, optional): Dimension of conformer encoder
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing input vector

    Returns: outputs
        - **outputs** (batch, time, dim): Tensor produces by conformer block.
    """
    def __init__(
            self,
            encoder_dim: int = 512,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
    ):
        super(ConbimambaBlock, self).__init__()
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1

        # 定义第一个FeedForward
        self.ResidualConn_A = ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            )

        # 定义外部双向Mamba
        self.ResidualConn_B = ResidualConnectionModule(
            module=ExBimamba(d_model=encoder_dim),
        )

        # 定义convolution层
        self.ResidualConn_C = ResidualConnectionModule(
                module=ConformerConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p,
                ),
            )

        # 定义第二个FeedForward
        self.ResidualConn_D = ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            )

        # 正则化
        self.norm = nn.LayerNorm(encoder_dim)


    def forward(self, inputs: Tensor) -> Tensor:

        x = self.ResidualConn_A(inputs) # 执行第一个Feed-Forward: (B,T,C)-->(B,T,C)
        x = self.ResidualConn_B(x) # 执行ExBimamba(外部双向Mamba): (B,T,C)-->(B,T,C)
        x = self.ResidualConn_C(x) # 执行Conformer的Convolution: (B,T,C)-->(B,T,C)
        x = self.ResidualConn_D(x) # 执行第二个Feed-Forward: (B,T,C)-->(B,T,C)
        out = self.norm(x) # 正则化: (B,T,C)-->(B,T,C)
        return out

# ConBi END

# sess_data BEGIN

class SessionDataset(Dataset):
    def __init__(self, train_data, test_data, user_key='user', item_key='item', session_key='sessionId',
                 time_key='timestamp', max_len_recent=10, device='cpu',
                 sample_size=1000, sampling='random', weighting='div'):
        print('init data loader')
        stime = time.time()
        self.item_key = item_key
        self.time_key = time_key
        self.user_key = user_key
        self.session_key = session_key

        # knn sampling
        self.sample_size = sample_size
        self.sampling = sampling
        self.weighting = weighting  # 计算neighbor session时item位置

        self.session_ids_train = train_data[session_key].unique()
        self.session_ids_test = test_data[session_key].unique()

        # ! 1.初始化重要的变量
        # session - item map
        self.session_item_map = {}  #: 映射session id到item set
        self.item_session_map = {}  # 映射item id到session id set
        # user - item map
        self.item_user_map = {}
        self.user_item_map = {}
        # session - user map
        self.session_user_map = {}  # 映射session id到user id
        self.user_session_map = {}  # not used
        # session - time map
        self.session_time = {}  # 映射session id到最后一个interaction的时间

        # session - set of recent clicked items map
        self.session_recent_map = {}
        self.recent_session_map = {}
        # user - set of recent items map
        self.user_recent_map = {}

        # self.last_n_days = last_n_days
        self.max_len_recent = max_len_recent  # recent items保存最近的item个数



        # ! 2. 扫描训练集
        print('scan the dataset')
        # train_data = train_data.copy()
        dataset = pd.concat((train_data, test_data))
        dataset.sort_values(
            by=[self.session_key, self.time_key], inplace=True)

        # get the position of the columns
        index_session = dataset.columns.get_loc(self.session_key)
        index_item = dataset.columns.get_loc(self.item_key)
        index_time = dataset.columns.get_loc(self.time_key)
        index_user = dataset.columns.get_loc(self.user_key)  # user_based

        self.max_length = dataset.groupby([self.session_key])[
            self.item_key].count().max()  # max length of session in train set

        self.itemids = train_data[self.item_key].unique()
        assert (len(np.setdiff1d(
            test_data[self.item_key].unique(), self.itemids, assume_unique=True)) == 0)
        self.userids = train_data[self.user_key].unique()
        assert (len(np.setdiff1d(
            test_data[self.user_key].unique(), self.userids, assume_unique=True)) == 0)
        # self.sessionids = train_data[self.session_key].unique()  # !只有训练集

        # ids start from 1, index 0 are padding items
        # map: item name to index
        self.item2id = dict(zip(self.itemids, range(1, len(self.itemids) + 1)))
        self.items = set(self.item2id.keys())
        # map: user name to index
        self.user2id = dict(zip(self.userids, range(1, len(self.userids) + 1)))
        # self.session2id = dict(zip(self.sessionids, range(1, len(self.sessionids)+1)))#map: session name to index
        self.id2item = dict()  # map: item index to name
        self.id2user = dict()  # map: user index to name
        for k in self.item2id.keys():
            self.id2item[self.item2id[k]] = k
        for k in self.user2id.keys():
            self.id2user[self.user2id[k]] = k

        self.user_number = len(self.userids)
        self.item_number = len(self.itemids)

        # 新增用户-物品频率统计

        self.user_item_freq = defaultdict(partial(defaultdict, int))
        for session in train_data.itertuples():
            self.user_item_freq[session[index_user]][session[index_item]] += 1

        # number of session in training set
        self.session_number = len(self.session_ids_train)
        print(f"users in training set:    {self.user_number}")
        print(f"items in training set:    {self.item_number}")
        print(f"sessions in training set: {self.session_number}")

        # ! 扫描训练集
        session = -1  # session id
        session_items = []  # 当前session的item集合
        recent_items = []  # 当前session对应user的历史clicked items,不包括当前session中的
        timestamp = -1  # session timestamp
        user = -1  # session user id
        for row in dataset.itertuples(index=False):
            # cache items of sessions
            if row[index_session] != session:  # 遇见新session id后的初始化
                # if len(session_items) > 0:  # 处理上一个session，收尾
                self.session_item_map.update(
                    {session: session_items})  # session包含的item
                # cache the last time stamp of the session
                self.session_time.update({session: timestamp})  # session时间
                self.session_user_map.update({session: user})  # session user
                # if time < self.min_time:
                #     self.min_time = time

                # 处理上一个session的recent items，对应旧session和旧user
                # 1. 根据user id获取user recent items
                recent_items = [0] * self.max_len_recent  # padding
                recent_items.extend(
                    self.user_recent_map.get(user, []))  # 将上一个处理的session内容更新到recent_items，对应上一个处理的session和user

                # 2. 截取最近的，作为session的recent items，同时记录recent item到session映射
                if self.max_len_recent > 0:
                    recent_items = recent_items[-self.max_len_recent:]
                self.session_recent_map.update(
                    {session: recent_items.copy()})  # session id到包含的item列表的映射
                for item in set(recent_items):  # 构造recent记录中item到session id的映射
                    if item != 0:
                        if item in self.recent_session_map:
                            self.recent_session_map[item].add(session)
                        else:
                            self.recent_session_map[item] = {session}

                # 3. 将session补充到user recent items
                recent_items.extend(session_items)
                self.user_recent_map.update(
                    {user: recent_items.copy()})  # user id到recent item的映射

                #! 新的session: 新user, session id
                user = self.user2id[row[index_user]]  # 初始化 user
                session = row[index_session]  # session id
                session_items = []  # 初始化session item
            timestamp = row[index_time]  # 更新session time
            item = self.item2id[row[index_item]]
            session_items.append(item)  # session item增加当前扫描到的

            # 更新item到session id的set的映射
            map_is = self.item_session_map.get(item)
            if map_is is None:
                map_is = set()
                self.item_session_map.update({item: map_is})
            map_is.add(row[index_session])

            # 更新item到user，以及user到item的映射，包含频度信息
            # user = self.user2id[row[index_user]]
            if item not in self.item_user_map:
                self.item_user_map[item] = {}
            self.item_user_map[item][user] = self.item_user_map[item].get(
                user, 0) + 1
            if user not in self.user_item_map:
                self.user_item_map[user] = {}
            self.user_item_map[user][item] = self.user_item_map[user].get(
                item, 0) + 1

        # Add the last tuple
        self.session_item_map.update({session: session_items})
        self.session_time.update({session: timestamp})
        self.session_user_map.update({session: user})  # user_based

        # 1. 根据user id获取user recent items
        recent_items = [0] * self.max_len_recent  # padding
        recent_items.extend(self.user_recent_map.get(user, []))

        # 2. 截取最近的，作为session的recent items，同时记录recent item到session映射
        if self.max_len_recent > 0:
            recent_items = recent_items[-self.max_len_recent:]
        self.session_recent_map.update(
            {session: recent_items.copy()})  # session id到包含的item列表的映射
        for item in set(recent_items):  # 构造recent记录中item到session id的映射
            if item != 0:
                if item in self.recent_session_map:
                    self.recent_session_map[item].add(session)
                else:
                    self.recent_session_map[item] = {session}

        # 3. 将session补充到user recent items
        recent_items.extend(session_items)
        self.user_recent_map.update(
            {user: recent_items.copy()})  # user id到recent item的映射

        #! 为测试集中的session，计算neighbor sessions
        print('prepare test set')
        self.neighbor_sessions = {}
        self.neighbor_recent_items = {}
        self.neighbor_userids = {}
        self.neighbor_similarity = {}
        self.items_to_boost = {}

        self.session_ids_test = [sid for sid in self.session_ids_test if sum(
            self.session_recent_map[sid]) > 0]
        # i = 0
        # l = []  # 统计item_to_predict中，target_item占多少
        # l2 = []
        # l3 = []
        for sess_id in tqdm(self.session_ids_test):
            # if i == 10:
            #     return
            # i += 1

            recent_items = self.session_recent_map[sess_id]
            session_items = self.session_item_map[sess_id]

            possible_neighbors = set()

            #! 1. recent items 根据session的recent items缩小neighbor session范围
            for item in set(recent_items):
                possible_neighbors = possible_neighbors | self.recent_session_map.get(
                    item, set())

            #! 2. context neighbors
            # for item in set(session_items):
            #     possible_neighbors = possible_neighbors | self.item_session_map.get(
            #         item, set())
            # possible_neighbors = possible_neighbors - \
            #     set(self.session_ids_test)

            #! 3. history items
            history_items = self.user_item_map[self.session_user_map[sess_id]]
            history_items = copy.deepcopy(history_items)
            for itm in session_items[1:]:
                if history_items[itm] > 1:
                    history_items[itm] = history_items[itm] - 1
                else:
                    history_items.pop(itm)

            # if sess_id in possible_neighbors:
            #     possible_neighbors.remove(sess_id)

            # 采样neighbor session，大小为sample_size
            if self.sample_size == 0:  # only target session
                result = set([sess_id])
            else:  # sample some sessions
                if len(possible_neighbors) > self.sample_size:

                    if self.sampling == 'recent':
                        sample = self.most_recent_sessions(
                            possible_neighbors, self.session_time[sess_id], self.sample_size)
                    elif self.sampling == 'random':
                        sample = random.sample(
                            possible_neighbors, self.sample_size)
                    else:
                        sample = possible_neighbors[:self.sample_size]

                    result = sample
                else:
                    result = possible_neighbors

            result = np.array(list(result))
            if self.sample_size > 0:  # 如果sample_size=0，只根据当前sess_id预测，如果>0需要去掉当前session_id
                result = result[result != sess_id]
            # consider target session's recent items, add ground truth items to item_to_predict set
            # result.append(sess_id)  # same user, same session context
            assert len(self.session_recent_map[sess_id]) > 0
            # session_ids = list(result)
            session_ids = [sid for sid in result if sum(
                self.session_recent_map[sid]) > 0]  # neighbor session只考虑有recent items的
            assert len(session_ids) > 0
            self.neighbor_sessions[sess_id] = session_ids
            self.neighbor_userids[sess_id] = torch.as_tensor([
                self.session_user_map[sid] for sid in session_ids], dtype=int, device=device)
            self.neighbor_recent_items[sess_id] = torch.as_tensor([
                self.session_recent_map[sid] for sid in session_ids], dtype=int, device=device)
            self.neighbor_similarity[sess_id] = self.calc_similarity(recent_items, session_ids)
            # set(self.session_item_map[sess_id])
            items_to_boost = set(history_items.keys())
            session_ids = np.array(session_ids)
            session_ids = session_ids[session_ids !=
                                      sess_id]  # 计算boost items不能泄露信息
            for neighbor in session_ids:
                items_to_boost = items_to_boost | set(
                    self.session_item_map[neighbor])

            # n_item_to_pred = 1000
            # if len(neighbor_items_to_predict) < n_item_to_pred:
            #     candidates = np.array(
            #         list(self.items - neighbor_items_to_predict))
            #     selected = np.random.permutation(
            #         candidates)[:(n_item_to_pred-len(neighbor_items_to_predict))]
            #     neighbor_items_to_predict = neighbor_items_to_predict | set(
            #         selected)
            # elif len(neighbor_items_to_predict) > n_item_to_pred:
            #     candidates = np.array(
            #         list(neighbor_items_to_predict-set(self.session_item_map[sess_id])))
            #     selected = np.random.permutation(candidates)[:(
            #         n_item_to_pred-len(self.session_item_map[sess_id]))]
            #     neighbor_items_to_predict = set(self.session_item_map[sess_id]) | set(
            #         selected)

            # neighbor_items_to_predict = self.items

            # targets = np.unique(session_items)[1:5]
            # l.extend(list(np.isin(targets, list(
            #     history_items.keys()))))
            # l2.extend(list(np.isin(targets, list(
            #     neighbor_items_to_predict))))
            # l3.extend(list(np.isin(targets, list(set(neighbor_items_to_predict) |
            #                                      set(history_items.keys())))))
            # self.neighbor_items_to_predict[sess_id] = torch.as_tensor(
            #     list(neighbor_items_to_predict), dtype=int, device=device)
            boost_items = np.zeros(len(self.items), dtype=int)
            boost_items[(np.array(list(items_to_boost))-1)] = 1
            self.items_to_boost[sess_id] = boost_items
        # print(np.mean(l))
        # print(np.mean(l2))
        # print(np.mean(l3))
        print(f'Finished, cost {int(time.time()-stime)} s.')

    def next_test_session(self, session_id, device='cpu'):
        """
        测试时，每次获得一个测试session的信息
        """
        # todo only test on short sessions, same with INSERT
        session_item = torch.as_tensor(
            self.session_item_map[session_id][:5], dtype=int, device=device)
        neighbor_users = self.neighbor_userids[session_id]
        neighbor_recent = self.neighbor_recent_items[session_id]
        neighbor_similarity = self.neighbor_similarity[session_id]
        items_to_boost = self.items_to_boost[session_id]

        return session_item, neighbor_users, neighbor_recent, neighbor_similarity, items_to_boost

    # def next_test_session2(self, session_id):
    #     """
    #     根据session context获取item to boost
    #     session_id : int
    #     return: (sess_len*n_item) 0-1 matrix
    #     """
    #     session_item = self.session_item_map[session_id][:5]
    #     neighbor_items_to_boost = []
    #
    #     possible_neighbors = set()
    #     for item in session_item[:-1]:
    #         # 结合recent item和context item获取的相关session
    #         possible_neighbors = possible_neighbors | self.item_session_map.get(
    #             item, set())
    #
    #         # avoid leak
    #         if session_id in possible_neighbors:
    #             possible_neighbors.remove(session_id)
    #
    #         # 采样neighbor session，大小为sample_size
    #         if self.sample_size == 0:  # only target session
    #             result = set([session_id])
    #         else:  # sample some sessions
    #             if len(possible_neighbors) > self.sample_size:
    #
    #                 if self.sampling == 'recent':
    #                     sample = self.most_recent_sessions(
    #                         possible_neighbors, self.session_time[session_id], self.sample_size)
    #                 elif self.sampling == 'random':
    #                     sample = random.sample(
    #                         possible_neighbors, self.sample_size)
    #                 else:
    #                     sample = possible_neighbors[:self.sample_size]
    #
    #                 result = sample
    #             else:
    #                 result = possible_neighbors
    #
    #         session_ids = list(result)
    #         # consider target session's recent items, add ground truth items to item_to_predict set
    #         # result.append(sess_id)  # same user, same session context
    #         # assert len(self.session_recent_map[session_id]) > 0
    #         # session_ids = list(result)
    #         # session_ids = [sid for sid in result if sum(
    #         #     self.session_recent_map[sid]) > 0]  # neighbor session只考虑有recent items的
    #         assert len(session_ids) > 0
    #
    #         items_to_boost = set()
    #         for neighbor in session_ids:
    #             items_to_boost = items_to_boost | set(
    #                 self.session_item_map[neighbor])
    #         boost_items = np.zeros(len(self.items), dtype=int)
    #         boost_items[(np.array(list(items_to_boost))-1)] = 1
    #         neighbor_items_to_boost.append(boost_items)
    #
    #     return np.concatenate(neighbor_items_to_boost).reshape(-1, len(self.items))

    def calc_similarity(self, recent_items, sessions):
        '''
        Calculates the configured similarity for the items in recent_items and each session in sessions.

        Parameters
        --------
        recent_items: set of item ids
        sessions: list of session ids

        Returns
        --------
        out : list of tuple (session_id,similarity)
        '''

        pos_map = {}  # item权重，根据位置
        length = len(recent_items)

        count = 1
        for item in recent_items:
            if self.weighting is not None:
                pos_map[item] = getattr(self, self.weighting)(count, length)
                count += 1
            else:
                pos_map[item] = 1

        items = set(recent_items)
        # neighbors = []
        similarities = []
        cnt = 0
        for session in sessions:  # 对于每个可能的相似session
            cnt = cnt + 1
            # get recent items of the session, look up the cache first
            # n_items = self.items_for_session(session) # 相似session的item set
            n_items = set(self.session_recent_map[session])

            # dot product
            # 计算session相似度，内积，但考虑每个共同item根据其在当前session中的位置
            similarity = self.vec(items, n_items, pos_map)
            if similarity > 0:
                # neighbors.append(session)
                similarities.append(similarity)
            else:
                # neighbors.append(session)
                similarities.append(1e-10)
        similarities = np.array(similarities)
        return similarities / (similarities.sum()+1e-20)

    def vec(self, first, second, map):
        '''
        Calculates the ? for 2 sessions

        Parameters
        --------
        first: Id of a session
        second: Id of a session

        Returns
        --------
        out : float value
        '''
        a = first & second
        sum = 0
        for i in a:
            sum += map[i]

        result = sum / len(map)

        return result

    def most_recent_sessions(self, sessions, target_timestamp, number):
        '''
        Find the most recent sessions in the given set

        Parameters
        --------
        sessions: set of session ids

        Returns
        --------
        out : set
        '''
        session_list = list(sessions)
        time_list = np.array(list(map(self.session_time.get, session_list)))
        rtime_list = time_list - target_timestamp  # 相对时间，正数是发生在目标时间之后
        sorted_sessions = np.array(session_list)[np.abs(rtime_list).argsort()]
        return sorted_sessions[:number]

    # def most_recent_sessions1(self, sessions, target_timestamp, number):
    #     '''
    #     Find the most recent sessions in the given set
    #
    #     Parameters
    #     --------
    #     sessions: set of session ids
    #
    #     Returns
    #     --------
    #     out : set
    #     '''
    #     sample = set()
    #
    #     tuples = list()
    #     for session in sessions:
    #         timestamp = self.session_time.get(session)  # ! ?这里不应该是
    #         if timestamp is None:
    #             print(' EMPTY stampSTAMP!! ', session)
    #         tuples.append((session, timestamp))
    #
    #     tuples = sorted(tuples, key=itemgetter(1), reverse=True)
    #     # print 'sorted list ', sortedList
    #     cnt = 0
    #     for element in tuples:
    #         cnt = cnt + 1
    #         if cnt > number:
    #             break
    #         sample.add(element[0])
    #     # print 'returning sample of size ', len(sample)
    #     return sample
    #
    # def possible_neighbor_sessions(self, input_item_id):
    #     '''
    #     Find a set of session to later on find neighbors in.
    #     A self.sample_size of 0 uses all sessions in which any item of the current session appears.
    #     self.sampling can be performed with the options "recent" or "random".
    #     "recent" selects the self.sample_size most recent sessions while "random" just choses randomly.
    #
    #     Parameters
    #     --------
    #     sessions: set of session ids
    #
    #     Returns
    #     --------
    #     out : set
    #     '''
    #
    #     # add relevant sessions for the current item
    #     self.relevant_sessions = self.relevant_sessions | self.sessions_for_item(
    #         input_item_id)
    #
    #     # if self.past_neighbors:  # user-based 未使用
    #     #     self.retrieve_past_neighbors(user_id)
    #
    #     if self.sample_size == 0:  # use all session as possible neighbors
    #
    #         # print('!!!!! runnig KNN without a sample size (check config)')
    #         possible_neighbors = self.relevant_sessions
    #
    #     else:  # sample some sessions
    #         if len(self.relevant_sessions) > self.sample_size:
    #
    #             if self.sampling == 'recent':
    #                 sample = self.most_recent_sessions(
    #                     self.relevant_sessions, self.sample_size)
    #             elif self.sampling == 'random':
    #                 sample = random.sample(
    #                     self.relevant_sessions, self.sample_size)
    #             else:
    #                 sample = self.relevant_sessions[:self.sample_size]
    #
    #             possible_neighbors = sample
    #         else:
    #             possible_neighbors = self.relevant_sessions
    #
    #     return possible_neighbors

    # def recent_neighbor_sessions(self, recent_items):
    #     '''
    #     根据recent items查找neighbor sessions
    #     给定当前session的recent列表
    #     对其中的每个item，查询recent列表中包含该item的session的id，依赖self.session_recent_map
    #     '''
    #     possible_neighbors = set()
    #
    #     for item in set(recent_items):
    #         possible_neighbors = possible_neighbors | self.recent_session_map.get(
    #             item, set())
    #
    #     if self.sample_size == 0:  # use all session as possible neighbors
    #
    #         # print('!!!!! runnig KNN without a sample size (check config)')
    #         result = possible_neighbors
    #
    #     else:  # sample some sessions
    #         if len(possible_neighbors) > self.sample_size:
    #
    #             if self.sampling == 'recent':
    #                 sample = self.most_recent_sessions(
    #                     possible_neighbors, self.sample_size)
    #             elif self.sampling == 'random':
    #                 sample = random.sample(
    #                     possible_neighbors, self.sample_size)
    #             else:
    #                 sample = possible_neighbors[:self.sample_size]
    #
    #             result = sample
    #         else:
    #             result = possible_neighbors
    #
    #     return result

    # def linear_score(self, i):
    #     return 1 - (0.1 * i) if i <= 100 else 0
    #
    # def same_score(self, i):
    #     return 1
    #
    # def div_score(self, i):
    #     return 1 / i
    #
    # def log_score(self, i):
    #     return 1 / (log10(i + 1.7))
    #
    # def quadratic_score(self, i):
    #     return 1 / (i * i)
    #
    # def linear(self, i, length):
    #     return 1 - (0.1 * (length - i)) if i <= 10 else 0
    #
    # def same(self, i, length):
    #     return 1
    #
    # def div(self, i, length):
    #     return i / length
    #
    # def log(self, i, length):
    #     return 1 / (log10((length - i) + 1.7))
    #
    # def quadratic(self, i, length):
    #     return (i / length) ** 2

    def __getitem__(self, index):
        session_id = self.session_ids_train[index]
        session_items = self.session_item_map[session_id]
        recent_items = self.session_recent_map[session_id]
        user_id = self.session_user_map[session_id]
        all_items = self.user_item_map[user_id].keys()
        return session_id, user_id, session_items, recent_items, all_items

    def __len__(self):
        return self.session_number


def collate_fn(sample_list, padding_idx=0, device='cpu'):  # , device=device):
    max_len_session_item = max([len(s[2]) for s in sample_list])
    max_len_recent_item = max([len(s[3]) for s in sample_list])

    r_user_id = []
    r_session_id = []
    r_session_item = []
    r_recent_item = []
    # r_session_item_len = []
    # r_recent_item_len = []
    r_all_item = []
    for sample in sample_list:
        session_len = len(sample[2])
        recent_len = len(sample[3])
        # padding
        session_items = [padding_idx] * (max_len_session_item - session_len)
        # np.random.shuffle(sample[2])  # todo 当session中item顺序不重要时
        # random_idx = np.random.permutation(range(len(sample[2])))
        # # random_idx = list(range(len(sample[2])))
        # tgt_item_idx = random_idx[-1]
        # sess = np.array(sample[2])[random_idx]
        sess = np.array(sample[2])
        session_items.extend(sess)

        recent_item = [padding_idx] * (max_len_recent_item - recent_len)
        recent_item.extend(sample[3])
        # mask掉recent items列表中的target item
        # 若target item在源session中的距离结尾位置小于recent长度，说明target item在recent中
        # if len(sample[2]) - tgt_item_idx < max_len_recent_item:
        #     mask = np.ones(max_len_recent_item)
        #     # mask的地方值为0，乘后为padding item0，其他位置为1，乘1后不变
        #     mask[-(len(sample[2]) - tgt_item_idx)] = 0
        #     recent_item = recent_item * mask

        all_item = list(sample[-1])

        r_user_id.append(sample[1])
        r_session_id.append(sample[0])
        r_session_item.append(session_items)
        r_recent_item.append(recent_item)
        # r_session_item_len.append(session_len)
        # r_recent_item_len.append(recent_len)
        r_all_item.append(torch.LongTensor(
            np.array(all_item)).to(device))  # list, not tensor
    return torch.LongTensor(np.array(r_user_id)).to(device), torch.LongTensor(np.array(r_session_item)).to(device), \
        torch.LongTensor(np.array(r_recent_item)).to(device), r_all_item


# def collate_fn2(sample_list, padding_idx=0, device='cpu'):
#     """
#     生成数据集：数据增强
#     对session x1,x2,...,xt生成x1:x2,x1,x2:x3,...
#
#     """
#     max_len_session_item = max([len(s[2]) for s in sample_list])
#     max_len_recent_item = max([len(s[3]) for s in sample_list])
#
#     r_user_id = []
#     r_session_id = []
#     r_session_item = []
#     r_recent_item = []
#     # r_session_item_len = []
#     # r_recent_item_len = []
#     r_all_item = []
#     for sample in sample_list:
#         session_len = len(sample[2])
#         recent_len = len(sample[3])
#         for i in range(2, session_len + 1):  # 2,3,...,
#             sess = sample[2][:i]
#             # padding
#             # session_items = [padding_idx] * (max_len_session_item - len(sess))
#             # session_items.append(sess)
#
#             # recent_item = [padding_idx] * (max_len_recent_item - recent_len)
#             # recent_item.append(sample[3])
#
#             all_item = list(sample[-1])
#
#             r_user_id.append(sample[1])
#             r_session_id.append(sample[0])
#             r_session_item.append(sess)
#             r_recent_item.append(sample[3])
#             r_all_item.append(all_item)  # list, not tensor
#     return r_user_id, r_session_item, r_recent_item, r_all_item

# sess_data END

# recommend BEGIN

class Recommender(nn.Module):
    def __init__(self, dim=64, num_user=100, num_item=100, b=0.1, dropout=0.1, device='cpu'):
        super(Recommender, self).__init__()

        self.user_embedding = nn.Embedding(
            num_user + 1, dim, padding_idx=0)
        self.user_embedding2 = nn.Embedding(
            num_user + 1, dim, padding_idx=0)  # logvar
        self.item_embedding = nn.Embedding(
            num_item + 1, dim, padding_idx=0)

        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

        # self.cweight = cweight
        self.b = b
        self.dim = dim
        self.device = device
        self.dropout = dropout
        self.num_item = num_item
        self.user_item_freq = defaultdict(lambda: defaultdict(int))
        self.num_negatives = 10

        self.user_encoder_mlp = nn.Linear(
            dim * 2, dim)
        self.predictor_mlp = nn.Linear(dim * 2, dim)
        self.predictor_mlp2 = nn.Linear(dim * 3, 1)

        self.gru = nn.GRU(dim, dim, batch_first=True)

        self.attn_pred = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 1)
        )
        self.attn_uencoder = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 1)
        )
        self.attn_cencoder = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 1)
        )

        self.conB = ConbimambaBlock(
            encoder_dim=dim,
            num_attention_heads=8,
            feed_forward_expansion_factor=2,
            conv_expansion_factor=2,
            feed_forward_dropout_p=0.1,
            attention_dropout_p=0.1,
            conv_dropout_p=0.1,
            conv_kernel_size=3,
            half_step_residual=True,
        )

        self.contrast_loss = ContrastiveLoss()

        # 替换原公式5的线性层为元网络
        self.meta_lambda = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

        self.mixer = MixerModel(d_model=64,
                           n_layer=2,
                           rms_norm=False,
                           drop_out_in_block=0.2,
                           drop_path=0.2,
                           training=True)

        # 新增动态增强权重网络
        self.boost_net = nn.Sequential(
            nn.Linear(2 * dim, dim),  # 输入：用户嵌入 + 物品嵌入
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()  # 输出增强权重（0~1）
        )
        self.boost_factor = nn.Parameter(torch.tensor(0.1))  # 可学习的增强系数


        self.__init_params()

    def __init_params(self):
        torch.nn.init.xavier_uniform_(self.user_embedding.weight[1:])
        torch.nn.init.xavier_uniform_(self.user_embedding2.weight[1:])
        torch.nn.init.xavier_uniform_(self.item_embedding.weight[1:])

    def get_boost_weights(self, user_emb, item_emb):
        """
        输入：
        - user_emb: [B, D] 用户嵌入
        - item_emb: [B, L, D] 近期交互物品嵌入
        输出：
        - boost_weights: [B, L] 每个物品的动态增强权重
        """
        B, L, D = item_emb.shape
        user_expanded = user_emb.unsqueeze(1).expand(B, L, D)  # [B, L, D]
        boost_input = torch.cat([user_expanded, item_emb], dim=-1)  # [B, L, 2D]
        boost_weights = self.boost_net(boost_input).squeeze(-1)  # [B, L]
        return boost_weights


    def forward(self, user_ids, sess_item, rcnt_item, items_to_predict=None):
        '''
        user_ids: get user embeddings for recent item encoder
        rcnt_item: user encoder input
        items_to_predict: B*n
        '''
        # items to score
        # print("User:", user_ids[0], user_ids[1])
        # print("rcnt_item:", rcnt_item[0], rcnt_item[1])
        # print("sess_item:", sess_item[0], sess_item[1])

        if items_to_predict is not None:
            # y_true = self.item_embedding(sess_item[:,-1]).unsqueeze(1) # B*1*dim : all target items
            if len(items_to_predict.size()) == 1:  # 每个用户用相同的items_to_predict
                vemb = self.item_embedding(
                    items_to_predict).unsqueeze(0)  # 1*(n_item)*dim
            elif len(items_to_predict.size()) == 2:
                vemb = self.item_embedding(
                    items_to_predict)  # 每个sample不同的候选item
            else:
                # print('error items to predict shape, predict scores of all items')
                vemb = self.item_embedding.weight[1:].unsqueeze(0)
            # predict scores
            # scores_true = self.predict(hu, hc, y_true)
            # scores_neg = self.predict(hu, hc, y_neg)

            # scores = torch.cat((scores_true, scores_neg), dim=1)
            # y_label = torch.zeros(y_neg.size(1)+1, dtype=int, device=self.user_embedding.weight.device)
            # scores = self.predict(hu, hc, vemb)
        else:
            vemb = self.item_embedding.weight[1:].unsqueeze(0)  # 1*n_item*dim
            # scores = self.predict(hu, hc, ) # B*(n_item)*dim: all items
            # y_label = sess_item[:,-1]
        # two encoders
        # hr = None#self.user_encoder(user_ids, rcnt_item)
        # hu = self.user_embedding(user_ids)  # self.user_encoder_test(user_ids, hr)  # B*dim


        hu = self.user_encoder7(user_ids, rcnt_item)  # B*dim

        hc = self.session_encoder(
            sess_item)  # B*sess_len*dim
        hc = self.mixer(hc)  # (B,L,D)-->(B,L,D)

        assert hu.isnan().sum() == 0
        assert hc.isnan().sum() == 0


        h, attn = self.predictor_attn(hu, hc, vemb)  # b*sess_len*n_item*dim

        # print(h.shape, attn.shape)



        scores = (h * vemb.unsqueeze(1)).sum(-1)  # b*sess_len*n_item

        assert scores.isnan().sum() == 0



        # --- 生成对比学习样本 ---
        item_emb = self.item_embedding(sess_item)  # [B, L, D]
        anchor = item_emb[:, :-1, :]  # 锚点：当前物品 [B, L-1, D]
        positive = item_emb[:, 1:, :]  # 正样本：下一物品 [B, L-1, D]

        # 负样本采样（每个锚点采样N个）
        B, L_minus_1, D = anchor.shape
        negative_indices = torch.randint(
            1, self.num_item + 1,
            (B, L_minus_1, self.num_negatives),  # 形状 [B, L-1, N]
            device=anchor.device
        )
        negative = self.item_embedding(negative_indices)  # [B, L-1, N, D]

        return scores, anchor, positive, negative, hu, hc, attn


    def test_session(self, user_ids, sess_item, rcnt_item, similarity):
        '''
        user_ids: get user embeddings for recent item encoder
        rcnt_item: user encoder input
        items_to_predict: B*n
        boost: n_item: one-hot vector, items to boost
        '''
        # items to score
        # if items_to_predict is not None:
        #     if len(items_to_predict.size()) == 1:  # 每个用户用相同的items_to_predict
        #         vemb = self.item_embedding(
        #             items_to_predict).unsqueeze(0)  # 1*(n_item)*dim
        #     elif len(items_to_predict.size()) == 2:
        #         vemb = self.item_embedding(
        #             items_to_predict)  # 每个sample不同的候选item
        #     else:
        #         vemb = self.item_embedding.weight[1:].unsqueeze(0)
        # else:
        vemb = self.item_embedding.weight[1:].unsqueeze(0)  # 1*n_item*dim

        hu = self.user_encoder7(user_ids, rcnt_item)  # B*dim

        hc = self.session_encoder(
            sess_item)  # 1*sess_len*dim
        hc = self.mixer(hc)  # (B,L,D)-->(B,L,D)

        assert hu.isnan().sum() == 0
        assert hc.isnan().sum() == 0

        h, attn = self.predictor_attn(hu, hc, vemb)  # b*sess_len*dim


        scores = (h * vemb.unsqueeze(1)).sum(-1)  # b*sess_len*n_item

        assert scores.isnan().sum() == 0

        # scores = torch.zeros_like(scores)  # todo

        # sess_len*n_item

        # --- 生成对比学习样本 ---
        item_emb = self.item_embedding(sess_item)  # [B, L, D]
        anchor = item_emb[:, :-1, :]  # 锚点：当前物品 [B, L-1, D]
        positive = item_emb[:, 1:, :]  # 正样本：下一物品 [B, L-1, D]

        # 负样本采样（每个锚点采样N个）
        B, L_minus_1, D = anchor.shape
        negative_indices = torch.randint(
            1, self.num_item + 1,
            (B, L_minus_1, self.num_negatives),  # 形状 [B, L-1, N]
            device=anchor.device
        )
        negative = self.item_embedding(negative_indices)  # [B, L-1, N, D]


        return (scores.detach().cpu() * similarity[:, np.newaxis, np.newaxis]).sum(0).softmax(-1).numpy()

    # def predictor_hc(self, hc, vemb):
    #     """
    #     input:
    #     hc,  session context embedding (b/1*sesslen*dim)
    #     vemb: b/1*n_item*dim
    #     output:
    #     b/1*sess_len*n_item*dim
    #     """
    #     s = hc.unsqueeze(2) * vemb.unsqueeze(1)
    #
    #     return s  # b * session_len * n_item * dim
    #
    # def predictor_hu(self, hu, vemb):
    #     """
    #     input:
    #     hu,  user embedding (b/1*dim)
    #     vemb: b/1*n_item*dim
    #     output:
    #     b/1*sess_len*n_item*dim
    #     """
    #     s = hu.unsqueeze(1).unsqueeze(1) * vemb.unsqueeze(1)
    #
    #     return s  # b * 1 * n_item * dim
    #
    # def predictor_attn2(self, hu, hc, vemb):
    #     """
    #     hu: b*dim
    #     hc: b/1*sess_len*dim
    #     vemb: b/1*n_item*dim
    #     """
    #     h_list = []
    #     batch_size = hu.size(0)
    #     for i in range(batch_size):
    #         hu_ = hu[i]  # dim
    #         hc_ = hc[i] if hc.size(0) == batch_size else hc[0]  # sess_len*dim
    #         vemb_ = vemb[i] if vemb.size(
    #             0) == batch_size else vemb[0]  # n_item*dim
    #
    #         # attention: b*sess_len
    #         hu_ = hu_.unsqueeze(0).unsqueeze(
    #             0).expand(hc.size(1), vemb.size(1), -1)
    #         hc_ = hc_.unsqueeze(1).expand_as(hu_)
    #         vemb_ = vemb_.unsqueeze(0).expand_as(hu_)  # sess_len*n_item*dim
    #         attn = torch.cat((hu_, hc_, vemb_), dim=-1)  # sess_len*n_item*3dim
    #         attn = F.dropout(attn, p=0.2, training=self.training)
    #         attn = self.predictor_mlp2(attn)
    #         attn = torch.sigmoid(attn)  # sess_len*1
    #         h = attn * hu_ + (1 - attn) * hc_  # sess_len*dim
    #         h_list.append(h)
    #
    #     return torch.cat(h_list)  # b*sess_len*nitem*dim
    #
    def predictor_attn(self, hu, hc, vemb):
        """
        hu: b*dim
        hc: b/1*sess_len*dim
        vemb: b*n_item*dim
        需要大内存
        """
        # attention: b*sess_len
        # print(hu.shape)
        # print(hc.shape)
        # print(vemb.shape)

        hu = hu.unsqueeze(1).unsqueeze(
            1).expand(-1, hc.size(1), vemb.size(1), -1)
        hc = hc.unsqueeze(2).expand_as(hu)
        vemb = vemb.unsqueeze(1).expand_as(hu)  # b*sess_len*n_item*dim

        # print(hu.shape)
        # print(hc.shape)
        # print(vemb.shape)


        # 需要大内存 b*sess_len*n_item*3dim
        attn = torch.cat((hu, hc, vemb), dim=-1)
        # print(1, attn.shape)
        attn = F.dropout(attn, p=0.2, training=self.training)
        # print(2, attn.shape)
        attn = self.predictor_mlp2(attn)
        # print(3, attn.shape)
        attn = torch.sigmoid(attn)  # b*sess_len*1
        # print(4, attn.shape)
        h = attn * hu + (1 - attn) * hc  # b*sess_len*dim

        # print(h.shape)
        # print(attn.shape)

        return h, attn  # b*sess_len*nitem*dim


    def predictor_attn1(self, hu, hc, vemb):
        """
        hu: b*dim
        hc: b/1*sess_len*dim
        vemb: b*n_item*dim
        需要大内存
        """
        # attention: b*sess_len


        hu = hu.unsqueeze(1).unsqueeze(
            1).expand(-1, hc.size(1), vemb.size(1), -1)
        hc = hc.unsqueeze(2).expand_as(hu)
        vemb = vemb.unsqueeze(1).expand_as(hu)  # b*sess_len*n_item*dim
        # # 需要大内存 b*sess_len*n_item*3dim

        # 动态拼接特征
        meta_input = torch.cat([
            hc,  # 会话上下文均值 [B, D]
            hu,  # 用户偏好 [B, D]
            vemb
            # self.user_embedding(users)  # 用户ID嵌入 [B, D]
        ], dim=-1)  # [B, 3D]


        # 生成动态λ
        lam = self.meta_lambda(meta_input)  # [B, 1]

        h = lam * hu + (1 - lam) * hc  # b*sess_len*dim

        return h, lam  # b*sess_len*nitem*dim

    # def predict_with_one(self, h, vemb):
    #     '''
    #
    #     测试，只使用h预测
    #     h: B*dim
    #     vemb: B*nitem*dim
    #     '''
    #     s = (h.unsqueeze(1) * vemb).sum(-1)
    #     # s = (h * vemb).sum(-1)
    #     # s = self.softmax_stable(s)
    #     return s

    # def predict_cat(self, hu, hc, q):
    #     """
    #     拼接hu hc，使用mlp映射再内积
    #
    #     hu:b*dim
    #     hc:b*dim
    #     q: b*nitem*dim
    #     """
    #     h = torch.cat((hu, hc), dim=-1)
    #     # h = F.dropout(h, p=0.2, training=self.training)
    #     h = self.predictor_mlp(h)
    #     # h = F.dropout(h, p=0.2, training=self.training)
    #     # h = torch.relu(h)
    #     q = q.expand(h.size(0), -1, h.size(-1))
    #     score = torch.bmm(h.unsqueeze(1), q.transpose(1, 2))  # b*1*nitem
    #     return score.squeeze(1)
    #
    # def predict_cat3(self, hu, hc, vemb):
    #     """
    #     拼接hu hc，使用mlp映射再内积
    #
    #     hu:b*dim
    #     hc:b*dim
    #     vemb: (b/1)*nitem*dim
    #     """
    #     hu = hu.unsqueeze(1).expand(hu.size(0), vemb.size(1), -1)
    #     hc = hc.unsqueeze(1).expand_as(hu)
    #     vemb = vemb.expand_as(hu)
    #     h = torch.cat((hu, hc, vemb), dim=-1)
    #     h = F.dropout(h, p=0.2, training=self.training)
    #     h = self.predictor_mlp2(h)
    #     h = torch.sigmoid(h)  # b*nitem*1
    #     h = h * hu + (1 - h) * hc  # b*nitem*dim
    #     score = (h * vemb).sum(-1)
    #     return score.squeeze(1)
    #
    # def predict(self, hu, hc, q):
    #     '''
    #     predict the scores of candidate items
    #     hu: B*dim: user preference embedding
    #     hc: B*dim: context embedding for sessions
    #     q: B*n_item*dim: embeddings of candidate items
    #     '''
    #     hu = hu.unsqueeze(1).unsqueeze(1)  # B*1*1*dim 第二维将hu和hc拼接，第三维对应n_item
    #     # predict next中，当前session context是唯一的，需要复制
    #     hc = hc.unsqueeze(1).unsqueeze(1).expand_as(hu)
    #     h = torch.cat([hu, hc], dim=1)  # B*2*1*dim: 合并hu hc，一起计算
    #
    #     h = F.relu(h)  # B*2*1*dim
    #     q = F.relu(q).unsqueeze(1)  # 1*1*n_item*dim
    #
    #     h = h.expand(-1, -1, q.size(2), -1)
    #     q = q.expand(h.size(0), 2, -1, -1)
    #
    #     # w = (h * q.unsqueeze(1)).sum(-1) # B*2*n_item 根据y计算hu和hc的权重
    #     # w = w/w.sum(1).unsqueeze(1)# B*2*n_item 两个权重和为1
    #     # B*2*n_item*1, 内积改为attention net
    #     w = self.attn_pred(torch.cat((h, q), dim=-1))
    #     # B*2*n_candidateitems*1
    #     w = torch.softmax(w - w.max(1, keepdim=True)[0].detach(), dim=1)
    #     # assert torch.isnan(w).sum() == 0
    #     h_fuse = (w * h).sum(1)  # B*n_item*dim, 将hu和hc加权求和
    #
    #     scores = (q[:, 0] * h_fuse).sum(-1)  # B*n_item
    #
    #     return scores

    # def softmax_stable(self, input, mask=None, dim=-1):
    #     '''
    #     stable softmax (no nan for exp) with mask (mask some entries)
    #     mask should be able to mul with input
    #     '''
    #     max_value = input.max(dim, keepdim=True)[0].detach()
    #     input = input - max_value
    #     input = input.exp() + 1e-10
    #     if mask is not None:
    #         input = input * mask
    #     sum_value = input.sum(dim, keepdim=True)
    #     return input / sum_value

    # def logsoftmax_stable(self, input, mask=None, dim=-1):
    #     input = self.softmax_stable(input, mask, dim)
    #     # input = input+1e-20
    #     return input.log()

    def session_encoder(self, sess_item, encoder_type='attn_last'):
        """
        session context encoder
        input:
        - sess_item: batch_size * session_len, LongTensor
        - encoder_type: string
        """
        if encoder_type == 'avg':
            mask = sess_item != 0  # B*sess_len
            h = self.item_embedding(sess_item).cumsum(1)
            h = h / (mask.cumsum(1).unsqueeze(-1)+1e-20)
        elif encoder_type == 'attn_avg':
            mask = sess_item != 0  # B*sess_len
            query = self.item_embedding(sess_item).cumsum(1)
            query = query / (mask.cumsum(1).unsqueeze(-1) +
                             1e-20)  # b*sess_len*dim
            query = self.mixer(query)  # (B,L,D)-->(B,L,D)

            key = value = self.item_embedding(sess_item)  # b*sess_len*dim
            key = self.mixer(key)
            value = self.mixer(value)
            attn_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
            h = self.step_attention(query, key, value, attn_mask)
        elif encoder_type == 'attn_last':
            mask = sess_item != 0  # B*sess_len
            query = self.item_embedding(sess_item)
            key = value = self.item_embedding(sess_item)
            attn_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
            h = self.step_attention(query, key, value, attn_mask)
        elif encoder_type == 'gru':
            sess_len = (sess_item != 0).sum(1)
            h0 = torch.zeros(1, sess_item.size(
                0), self.dim, device=self.device)
            hs = F.dropout(self.item_embedding(sess_item), p=self.dropout)
            # if isinstance(sess_len, np.ndarray):
            #     sess_len = torch.LongTensor(sess_len)
            hs = torch.nn.utils.rnn.pack_padded_sequence(
                hs, sess_len.to('cpu'), batch_first=True, enforce_sorted=False)
            hs, h0 = self.gru(hs, h0)
            hs, _ = torch.nn.utils.rnn.pad_packed_sequence(
                hs, batch_first=True)
            h = hs
        elif encoder_type == 'gru_res':
            sess_len = (sess_item != 0).sum(1)
            h0 = torch.zeros(1, sess_item.size(
                0), self.dim, device=self.device)
            hs = F.dropout(self.item_embedding(sess_item), p=self.dropout)
            # if isinstance(sess_len, np.ndarray):
            #     sess_len = torch.LongTensor(sess_len)
            hs = torch.nn.utils.rnn.pack_padded_sequence(
                hs, sess_len.to('cpu'), batch_first=True, enforce_sorted=False)
            hs, h0 = self.gru(hs, h0)
            hs, _ = torch.nn.utils.rnn.pad_packed_sequence(
                hs, batch_first=True)
            h = hs + self.item_embedding(sess_item)  # 残差
        else:
            print(f'not supported encoder_type: {encoder_type}')
        return h

    def step_attention(self, query, key, value, mask=None):
        """
        step attention for session:

        Args: dim, mask
            dim (int): dimention of attention
            mask (torch.Tensor): tensor containing indices to be masked
        Inputs: query, key, value, mask
            - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
            - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
            - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
            - **mask** (-): tensor containing indices to be masked
        Returns: context, attn
            - **context**: tensor containing the context vector from attention mechanism.
            - **attn**: tensor containing the attention (alignment) from the encoder outputs.
        """
        sqrt_dim = np.sqrt(query.size(-1))
        # score: b*d1*d2 d1是query数量，d2是key数量
        score = torch.bmm(query, key.transpose(1, 2)) / sqrt_dim

        # softmax
        score = score.exp()  # the exp in softmax
        if mask is not None:  # mask的地方为true,能够view为b*d1*d2
            # score.masked_fill_(mask, 0)
            score = score*mask
        h = score.unsqueeze(-1) * value.unsqueeze(1)  # b*d1*d2*dim
        h = h.cumsum(2) / (score.cumsum(2).unsqueeze(-1)+1e-20)  # norm for d2
        h = h.sum(2)  # weighted sum
        assert torch.sum(torch.isnan(h)) == 0
        return h  # B*d1*dim

    def SDPattention(self, query, key, value, mask=None):
        """
        ref: https://github.com/sooftware/attentions/blob/master/attentions.py
        Scaled Dot-Product Attention proposed in "Attention Is All You Need"
        Compute the dot products of the query with all keys, divide each by sqrt(dim),
        and apply a softmax function to obtain the weights on the values
        Args: dim, mask
            dim (int): dimention of attention
            mask (torch.Tensor): tensor containing indices to be masked
        Inputs: query, key, value, mask
            - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
            - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
            - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
            - **mask** (-): tensor containing indices to be masked
        Returns: context, attn
            - **context**: tensor containing the context vector from attention mechanism.
            - **attn**: tensor containing the attention (alignment) from the encoder outputs.
        """
        sqrt_dim = np.sqrt(query.size(-1))
        score = torch.bmm(query, key.transpose(1, 2)) / \
            sqrt_dim  # b*d1*d2 d1是query数量，d2是key数量
        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))
        attn = torch.softmax(score, -1)
        context = torch.bmm(attn, value)  # b*d1*dim
        return context

    # def loss_predict(self, scores, targets):
    #     '''
    #
    #     '''
    #     loss = nn.NLLLoss()
    #     # a = nn.LogSoftmax(dim=1)
    #     scores = self.softmax_stable(scores)
    #     if scores.min() == 0:
    #         scores = scores + 1e-20
    #     scores = scores.log()
    #     return loss(scores, targets)

    def loss_predict1(self, scores, sess_item, map_target):
        """
        scores: b*sess_len*n_item
        sess_item: b*sess_len
        targets: n_item_to_score
        map_target: b*sess_len
        """
        loss = nn.CrossEntropyLoss(ignore_index=0)
        ctx_item = sess_item[:, :-1].reshape(-1)  # b*(sess_len-1)
        tgt_item = map_target[:, 1:].reshape(-1)

        # mask padding items
        mask = tgt_item != 0
        scores = scores.view(-1, scores.size(-1))[mask]
        tgt_item = tgt_item[mask]

        return loss(scores, tgt_item)

    # def loss_predict2(self, scores, targets):
    #     '''
    #     first items are true label, others are negative label
    #     '''
    #     loss = FocalLoss(gamma=5)
    #     return loss(scores, targets)

    # def loss_bpr(self, logit):
    #     """
    #     logit: B*B*dim
    #     ref: https://github.com/hungthanhpham94/GRU4REC-pytorch/blob/666b84264c4afae757fe55c6997dcf0a4da1d44e/lib/lossfunction.py
    #     """
    #     diff = logit.diag().view(-1, 1).expand_as(logit) - logit
    #     loss = -torch.mean(F.logsigmoid(diff))
    #     return loss
    #
    # def loss_top1(self, logit):
    #     """
    #     logit: B*B*dim
    #     ref: https://github.com/hungthanhpham94/GRU4REC-pytorch/blob/666b84264c4afae757fe55c6997dcf0a4da1d44e/lib/lossfunction.py
    #     """
    #     diff = -(logit.diag().view(-1, 1).expand_as(logit) - logit)
    #     loss = torch.sigmoid(diff).mean() + torch.sigmoid(logit ** 2).mean()
    #     return loss

    # def loss_causal(self, scores, sess_ctx, usr_hist, targets):
    #     """
    #     scores: batch_size*(sess_len-1)*num_item*1
    #     sess_ctx: batch_size*(sess_len-1)
    #     usr_hist: list of long tensor
    #     torgets: num_item long tensor
    #     """
    #     scores = scores.squeeze(-1)
    #
    #     yc = torch.cat([torch.isin(targets, c).view(1, -1)
    #                    for c in sess_ctx])  # b*num_item
    #     yu = torch.cat([torch.isin(targets, h).view(1, -1)
    #                    for h in usr_hist])  # b*num_item
    #
    #     print(yc.shape)
    #     print(yu.shape)
    #     yc = yc.unsqueeze(1).expand(-1, scores.size(1), -1)
    #     yu = yu.unsqueeze(1).expand(-1, scores.size(1), -1)
    #     print(yc.shape)
    #     print(yu.shape)
    #
    #     mask = (sess_ctx != 0).view(-1)  # (batch_size*sess_len)
    #     print("mask:", mask.shape)
    #     yc = yc.reshape(-1, yc.size(-1)).float()[mask]
    #     yu = yu.reshape(-1, yu.size(-1)).float()[mask]
    #     scores = scores.reshape(-1, scores.size(-1))[mask]
    #     print(yc.shape)
    #     print(yu.shape)
    #     print(scores.shape)
    #
    #     bce1 = nn.BCELoss()
    #     bce2 = nn.BCELoss()
    #
    #     return bce1(scores, yu) + bce2(1-scores, yc)


    def loss_causal1(self, scores, sess_ctx, usr_hist, targets, user_ids):
        """
        scores: batch_size*(sess_len-1)*num_item*1
        sess_ctx: batch_size*(sess_len-1)
        usr_hist: list of long tensor
        torgets: num_item long tensor
        """
        scores = scores.squeeze(-1)

        batch_size = scores.size(0)
        num_items = scores.size(-1)  # 总物品数

        # --- 1. 获取频率权重（一维张量）---
        freq_weights = torch.zeros(batch_size, dtype=torch.float32, device=scores.device)
        for i in range(batch_size):
            user = user_ids[i].item()  # 当前用户ID
            target_item = targets[i].item()  # 当前目标物品ID
            freq = self.user_item_freq.get(user, {}).get(target_item, 0)
            freq_weights[i] = freq

        # 归一化频率权重（按整个批次）
        max_freq = freq_weights.max()
        freq_weights = freq_weights / (max_freq + 1e-10)  # [B]

        # --- 2. 生成加权伪标签 ---
        # (a) OSC伪标签（用户是否在历史中点击过目标物品）
        # (b) ISC伪标签（目标物品是否在会话上下文中）
        yc = torch.cat([torch.isin(targets, c).view(1, -1)
                        for c in sess_ctx])  # b*num_item
        yu = torch.cat([torch.isin(targets, h).view(1, -1)
                        for h in usr_hist])  # b*num_item
        freq_weights = freq_weights.unsqueeze(1).expand(-1, yu.size(1))

        freq_weights = freq_weights.unsqueeze(1).expand(-1, scores.size(1), -1)
        yc = yc.unsqueeze(1).expand(-1, scores.size(1), -1)
        yu = yu.unsqueeze(1).expand(-1, scores.size(1), -1)

        mask = (sess_ctx != 0).view(-1)  # (batch_size*sess_len)

        # --- 3. 计算二元交叉熵损失 ---
        # 提取模型预测分数中对应目标位置的 logit

        yc = yc.reshape(-1, yc.size(-1)).float()[mask]
        yu = yu.reshape(-1, yu.size(-1)).float()[mask]
        freq_weights = freq_weights.reshape(-1, freq_weights.size(-1))[mask]

        bce1 = nn.BCELoss()
        bce2 = nn.BCELoss()

        return bce1(freq_weights, yu) + bce2(1 - freq_weights, yc)



    # def loss_contrastive(self, all_item, sess_item, hu, hc):
    #     '''
    #     self-supervised loss
    #     hc: b*sess_len*dim
    #     hu: b*dim
    #     '''
    #     # label user preference: average of embedding of items the user clicked
    #     y_all_item = torch.cat([self.item_embedding(uv).mean(
    #         0).unsqueeze(0) for uv in all_item], 0)  # B*dim
    #     # label context: average of embedding of items in the session context
    #     mask = sess_item != 0  # B*(sess_len-1)
    #     y_ctx_item = self.item_embedding(sess_item).cumsum(
    #         1) / (mask.cumsum(1).unsqueeze(-1)+1e-20)  # B*sess_len*dim
    #
    #     y_all_item = y_all_item.detach()
    #     y_ctx_item = y_ctx_item.detach()
    #
    #     # 方式1: inner product
    #     loss = ((F.softplus((hc * y_all_item.unsqueeze(1)).sum(
    #         -1) - (hc * y_ctx_item).sum(-1)))*mask).mean()  # hu close to y_all_item than y_ctx_item
    #     loss = loss + (F.softplus(
    #         (hu.unsqueeze(1) * y_ctx_item).sum(-1) - (hu * y_all_item).sum(-1).unsqueeze(1))*mask).mean()  # hc close to avg_rc
    #
    #     # loss = (torch.sigmoid((hc * y_ctx_item).sum(-1) - (hc * y_all_item).sum(
    #     #     -1)) + 1e-10).log().mean()  # hu close to y_all_item than y_rct_item
    #     # loss = loss + (torch.sigmoid(
    #     #     (hu * y_all_item).sum(-1) - (hu * y_ctx_item).sum(-1)) + 1e-10).log().mean()
    #     # 方式2: Eucilidean distance
    #     # loss = F.triplet_margin_loss(hu, y_all_item, y_ctx_item, margin=self.margin)
    #     # loss = loss + F.triplet_margin_loss(hc, y_ctx_item, y_all_item, margin=self.margin)
    #     # assert torch.sum(torch.isnan(loss)) == 0
    #     return loss

    # def user_encoder(self, user_ids, rcnt_item):
    #     q = self.user_embedding(user_ids)  # B*dim: user embedding as query
    #     # B*rcnt_len*dim: recent item embeddings as key and value
    #     kv = self.item_embedding(rcnt_item)
    #     mask = rcnt_item != 0
    #     # a = (q.unsqueeze(1) * kv).sum(-1).exp() * mask # B*(rcnt_len), exp result nan
    #     a = self.attn_uencoder(torch.cat(
    #         (q.unsqueeze(1).expand_as(kv), kv), dim=-1)).squeeze(-1)  # B*rcnt_len
    #     a = a - a.max(1, keepdim=True)[0].detach()
    #     a = a.exp() * mask
    #     a = a / (a.sum(1).unsqueeze(1) + 1e-10)
    #     # assert torch.isnan(a).sum() == 0
    #     h = a.unsqueeze(-1) * kv  # B*(rcnt_len)*dim: attention weights * value
    #     return h.sum(1)  # B*dim
    #
    # def user_encoder2(self, user_ids):
    #     """
    #     仿造vae做法，学习用户embedding的mean和std，再采样
    #
    #     """
    #     # reparameterization
    #     mu = self.user_embedding(user_ids)
    #     log_var = self.user_embedding2(user_ids)
    #     sigma = torch.exp(log_var * 0.5)
    #     eps = torch.randn_like(sigma)
    #
    #     # loss
    #     KLD = 0.5 * torch.sum(torch.exp(log_var) +
    #                           torch.pow(mu, 2) - 1. - log_var)
    #     # return mu, KLD
    #
    #     return mu + sigma * eps, KLD
    #
    # def user_encoder3(self, rcnt_items):
    #     """
    #     最近item的平均作为user embedding
    #
    #     """
    #     mask = rcnt_items != 0  # B*sess_len
    #     h = self.item_embedding(rcnt_items).sum(
    #         1) / mask.sum(1, keepdim=True)  # B*dim
    #     return h  # B*dim

    # def user_encoder4(self, user_ids, rcnt_items):
    #     """
    #     使用user embedding作为query，内积attention
    #     """
    #     mask = rcnt_items != 0  # b*nitem
    #     q = self.user_embedding(user_ids)  # b*dim
    #     kv = self.item_embedding(rcnt_items)  # b*nitem*dim
    #     a = self.softmax_stable(
    #         (q.unsqueeze(1) * kv).sum(-1, keepdim=True), mask.unsqueeze(-1), 1)  # b*nitem*1
    #     h = (a * kv).sum(1)  # b*dim
    #     return h

    # def user_encoder5(self, user_ids, rcnt_items):
    #     """
    #     使用scaled dot product attention
    #     user emb作为query
    #     item in recent 作为key和value
    #     """
    #     mask = (rcnt_items == 0).unsqueeze(
    #         1)  # b*1*nitem mask kv的位置（第3维，第2是query的，见attention内计算score维度）
    #     q = self.user_embedding(user_ids).unsqueeze(1)  # b*1*dim
    #     kv = self.item_embedding(rcnt_items)  # b*nitem*dim
    #     h = self.SDPattention(q, kv, kv, mask)  # b*1*dim
    #     return h.squeeze(1)

    # def user_encoder6(self, user_ids, hr):
    #     """
    #     user embedding作为hu
    #     """
    #     # q = self.user_embedding(user_ids).unsqueeze(1) # B*1
    #
    #     # kv = self.item_embedding(rcnt_item) # B*nitem*dim
    #     q = self.user_embedding(user_ids)  # B*dim
    #     # h = torch.cat((q, hr), dim=-1)
    #     # h = self.user_encoder_mlp(h)
    #     # result = (q+hr)
    #     return q

    def user_encoder7(self, user_ids, rcnt_items):
        """
        使用scaled dot product attention
        使用用户的user embedding作为query
        item 作为key和value
        sess_item: b*len_sess
        """
        q = self.user_embedding(user_ids)  # b*dim
        # b*nitem mask kv的位置（第3维，第2是query的，见attention内计算score维度）

        # q = q.unsqueeze(1)
        # q = self.conB(q)
        # q = q.squeeze(1)

        mask = (rcnt_items == 0)  # b*n_recent


        kv = self.item_embedding(rcnt_items)  # b*nitem*dim

        kv = self.conB(kv)

        h = self.SDPattention(q.unsqueeze(1), kv, kv,
                              mask.unsqueeze(1))  # b*1*dim
        return h.squeeze(1)

    # def user_encoder8(self, user_ids, rcnt_items):
    #     """
    #     同user encoder7, 但残差
    #     item 作为key和value
    #     sess_item: b*len_sess
    #     """
    #     q = self.user_embedding(user_ids)  # b*dim
    #     # b*nitem mask kv的位置（第3维，第2是query的，见attention内计算score维度）
    #     mask = (rcnt_items == 0)  # b*n_recent
    #     kv = self.item_embedding(rcnt_items)  # b*nitem*dim
    #     h = self.SDPattention(q.unsqueeze(1), kv, kv,
    #                           mask.unsqueeze(1))  # b*1*dim
    #     return h.squeeze(1) + q

    # def ctxt_encoder(self, sess_item):
    #     '''
    #     context item embedding的平均
    #     sess_item: batch_size * context_len
    #     '''
    #     mask = sess_item != 0  # B*sess_len
    #     h = self.item_embedding(sess_item).cumsum(1)
    #     h = h / mask.cumsum(1).unsqueeze(-1)
    #     return h  # B*context_len*dim
    #
    # def ctxt_encoder1(self, sess_item):
    #     '''
    #
    #     以context item embedding的平均作为query
    #     '''
    #     mask = sess_item != 0  # B*sess_len
    #     q = self.item_embedding(sess_item).sum(
    #         1) / mask.sum(1, keepdim=True)  # B*dim
    #     # q = self.item_embedding(sess_item[:, -1])  # B*dim last item of context as query
    #     # B*(sess_len-1)*dim all context item as key and value
    #     kv = self.item_embedding(sess_item)
    #     # mask = sess_item != 0  # B*(sess_len-1): mask padding item
    #     # a = (q.unsqueeze(1) * kv).sum(-1).exp() * mask # B*(sess_len-1), exp result nan
    #     a = (q.unsqueeze(1) * kv).sum(-1, keepdim=True)  # B*sess_len*1
    #     a = self.softmax_stable(a, mask.unsqueeze(-1), 1)
    #     h = (a * kv).sum(1)
    #     return h
    #
    #     # a = self.attn_cencoder(torch.cat((q.unsqueeze(1).expand_as(kv), kv), dim=-1)).squeeze(-1)
    #     # a = a - a.max(1, keepdim=True)[0].detach()  # B*max_sess_len-1
    #     # a = a.exp() * mask
    #     # # assert torch.isnan(a).sum() == 0
    #     # a = a / (a.sum(1).unsqueeze(1) + 1e-10)
    #     # h = a.unsqueeze(-1) * kv  # B*(sess_len-1)*dim: attention weights * value
    #     # return h.sum(1)  # B*dim
    #
    # def ctxt_encoder2(self, sess_item):
    #     '''
    #     last item作为query
    #     '''
    #     q = self.item_embedding(
    #         sess_item[:, -1])  # B*dim last item of context as query
    #     # B*(sess_len-1)*dim all context item as key and value
    #     kv = self.item_embedding(sess_item)
    #     mask = sess_item != 0  # B*(sess_len-1): mask padding item
    #     # a = (q.unsqueeze(1) * kv).sum(-1).exp() * mask # B*(sess_len-1), exp result nan
    #     a = self.attn_cencoder(
    #         torch.cat((q.unsqueeze(1).expand_as(kv), kv), dim=-1)).squeeze(-1)
    #     a = a - a.max(1, keepdim=True)[0].detach()  # B*max_sess_len-1
    #     a = a.exp() * mask
    #     # assert torch.isnan(a).sum() == 0
    #     a = a / (a.sum(1).unsqueeze(1) + 1e-10)
    #     # B*(sess_len-1)*dim: attention weights * value
    #     h = a.unsqueeze(-1) * kv
    #     return h.sum(1)  # B*dim
    #
    # def ctxt_encoder3(self, sess_item):
    #     '''
    #
    #     context item embedding的平均
    #     '''
    #     mask = sess_item != 0  # B*sess_len
    #     h = self.item_embedding(sess_item).sum(
    #         1) / mask.sum(1, keepdim=True)  # B*dim
    #     return h  # B*dim
    #
    # def ctxt_encoder4(self, sess_item, user_ids):
    #     '''
    #     使用user id计算每个item的权重，再平均
    #     context item embedding的平均
    #     '''
    #     uemb = self.user_embedding(user_ids)  # b*dim
    #     vemb = self.item_embedding(sess_item)  # b*sesslen*dim
    #     mask = (sess_item != 0).unsqueeze(-1)  # b*sesslen
    #     a = (uemb.unsqueeze(1) * vemb).sum(-1).unsqueeze(-1)  # b*sesslen*1
    #     a = self.softmax_stable(a, mask, 1)  # b*sesslen*1
    #     # a = a - a.max(1, keepdim=True)[0].detach()  # B*max_sess_len*1
    #     # a = a.exp() * mask.unsqueeze(-1)
    #     # a = a/a.sum(1, keepdim=True)
    #
    #     return (a * vemb).sum(1)  # B*dim
    #
    # def ctxt_encoder5(self, sess_item, vemb):
    #     """
    #     ! out of memory error
    #     根据target item构造对应的context embedding，产生的维度比其他的encoder高
    #     sess_item: B*sess_len
    #     vemb: B*num_item*dim
    #
    #     return:
    #     h: B*num_item_to_predict*dim
    #     """
    #     ctx_emb = self.item_embedding(sess_item)  # B*sess_len*dim
    #     a = (ctx_emb.unsqueeze(2) * vemb.unsqueeze(1)
    #          ).sum(-1, keepdim=True)  # B*sess_len*num_item*1
    #     mask = sess_item != 0  # B*sess_len
    #     # B*sess_len*num_item*1
    #     a = self.softmax_stable(a, mask.unsqueeze(-1).unsqueeze(-1), dim=1)
    #     h = (a * ctx_emb.unsqueeze(2)).sum(1)  # B*num_item*dim
    #     return h
    #
    # def ctxt_encoder6(self, sess_item):
    #     """
    #     使用scaled dot product attention
    #     使用用户的context items的平均作为query
    #     item 作为key和value
    #     sess_item: b*len_sess
    #     """
    #     mask = (sess_item !=
    #             0)  # b*nitem mask kv的位置（第3维，第2是query的，见attention内计算score维度）
    #     q = self.item_embedding(sess_item).sum(
    #         1) / mask.sum(1).view(-1, 1)  # b*dim
    #     kv = self.item_embedding(sess_item)  # b*nitem*dim
    #     h = self.SDPattention(q.unsqueeze(1), kv, kv, ~
    #                           mask.unsqueeze(1))  # b*1*dim
    #     return h.squeeze(1)

# recommend END