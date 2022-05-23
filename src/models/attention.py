# -*- coding: utf-8 -*-
# pylint: disable-msg=import-error
# pylint: disable-msg=too-many-ancestors
# pylint: disable-msg=arguments-differ
# ========================================================
"""This module is written for write attentions classes."""
# ========================================================


# ========================================================
# Imports
# ========================================================

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as function


# Pytorch Implementation of some attention
# any questions, bug reports or recommends, please contacts sh951011@gmail.com
class SimpleAttention(nn.Module):
    """
    In this class we implement Attention model for text classification
    """

    def __init__(self, rnn_size):
        super(SimpleAttention, self).__init__()
        self.att_fully_connected_layers1 = nn.Linear(rnn_size, 350)
        self.att_fully_connected_layers2 = nn.Linear(350, 30)

    '''
    input param: 
        lstm_output: output of bi-LSTM (batch_size, sent_len, hid_dim * num_directions)   
    return: 
        attention weights matrix, 
        attention_output(batch_size, num_directions * hidden_size): attention output vector for the batch
        with this informations:
        r=30 and da=350 and penalize_coeff = 1
    '''

    def forward(self, lstm_output):
        # work_flow: lstm_output>fc1>tanh>fc2>softmax
        # usage of softmax is to calculate the distribution probability through one sentence :)
        lstm_output = lstm_output.permute(1, 0, 2)
        hidden_matrix = self.att_fully_connected_layers2(torch.tanh(self.att_fully_connected_layers1(lstm_output)))
        # hidden_matrix.size() :(batch_size, sent_len, num_head_attention)===> torch.Size([64, 150, 30])
        # for each of this 150 word, we have 30 feature from 30 attention's head.
        # permute? because the softmax will apply in 3rd dimension and we want apply it on sent_len so:
        hidden_matrix = hidden_matrix.permute(0, 2, 1)
        # hidden_matrix.size() :(batch_size, num_head_attention, sent_len)===>torch.Size([64, 30, 150])
        hidden_matrix = function.softmax(hidden_matrix, dim=2)
        # hidden_matrix.size() :(batch_size, num_head_attention, sent_len)===>torch.Size([64, 30, 150])
        # print(hidden_matrix.size())

        return hidden_matrix


class MultiHeadedLocationAwareAttention(nn.Module):
    r"""
    Applies a multi-headed location-aware attention mechanism on the output features from the decoder.
    Location-aware attention proposed in "Attention-Based Models for Speech Recognition" paper.
    The location-aware attention mechanism is performing well in speech recognition tasks.
    In the above paper applied a signle head, but we applied multi head concept.
    Args:
        hidden_dim (int): The number of expected features in the output
        num_heads (int): The number of heads. (default: )
        conv_out_channel (int): The number of out channel in convolution
    Inputs: query, value, prev_attn
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **prev_attn** (batch_size * num_heads, v_len): tensor containing previous timestep`s attention (alignment)
    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the feature from encoder outputs
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoder outputs.
    Reference:
        - **Attention Is All You Need**: https://arxiv.org/abs/1706.03762
        - **Attention-Based Models for Speech Recognition**: https://arxiv.org/abs/1506.07503
        - **State-Of-The-Art Speech Recognition with Sequence-to-Sequence Models**: https://arxiv.org/abs/1712.01769
    """

    def __init__(self, hidden_dim, num_heads=8, conv_out_channel=10):
        super(MultiHeadedLocationAwareAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = int(hidden_dim / num_heads)
        self.loc_projection = nn.Linear(conv_out_channel, self.dim, bias=False)
        self.loc_conv = nn.Conv1d(num_heads, conv_out_channel, kernel_size=3, padding=1)
        self.query_projection = nn.Linear(hidden_dim, self.dim * num_heads, bias=False)
        self.value_projection = nn.Linear(hidden_dim, self.dim * num_heads, bias=False)
        self.score_projection = nn.Linear(self.dim, 1, bias=True)
        self.out_projection = nn.Linear(hidden_dim << 1, hidden_dim, bias=True)
        self.bias = nn.Parameter(torch.rand(self.dim).uniform_(-0.1, 0.1))

    def forward(self, query, value, prev_attn):
        batch_size, seq_len = value.size(0), value.size(1)
        residual = query

        # Initialize previous attn (alignment) to zeros
        if prev_attn is None:
            prev_attn = value.new_zeros(batch_size, self.num_heads, seq_len)

        # Calculate location energy
        loc_energy = torch.tanh(self.loc_projection(self.loc_conv(prev_attn).transpose(1, 2)))  # BxNxT => BxTxD
        loc_energy = loc_energy.unsqueeze(1).repeat(1, self.num_heads, 1, 1).view(-1, seq_len,
                                                                                  self.dim)  # BxTxD => BxNxTxD

        # Shape matching
        query = self.query_projection(query).view(batch_size, -1, self.num_heads, self.dim).permute(0, 2, 1,
                                                                                                    3)  # Bx1xNxD
        value = self.value_projection(value).view(batch_size, -1, self.num_heads, self.dim).permute(0, 2, 1,
                                                                                                    3)  # BxTxNxD
        query = query.contiguous().view(-1, 1, self.dim)  # BNx1xD
        value = value.contiguous().view(-1, seq_len, self.dim)  # BNxTxD

        # Get attention score, attn
        score = self.score_projection(torch.tanh(value + query + loc_energy + self.bias)).squeeze(2)  # BNxT
        attn = seq_len.softmax(score, dim=1)  # BNxT

        value = value.view(batch_size, seq_len, self.num_heads, self.dim).permute(0, 2, 1, 3)  # BxTxNxD => BxNxTxD
        value = value.contiguous().view(-1, seq_len, self.dim)  # BxNxTxD => BNxTxD

        # Get context vector
        context = torch.bmm(attn.unsqueeze(1), value).view(batch_size, -1,
                                                           self.num_heads * self.dim)  # BNx1xT x BNxTxD => BxND
        attn = attn.view(batch_size, self.num_heads, -1)  # BNxT => BxNxT

        # Get output
        combined = torch.cat([context, residual], dim=2)
        output = self.out_projection(combined.view(-1, self.hidden_dim << 1)).view(batch_size, -1, self.hidden_dim)

        return output, attn


class LocationAwareAttention(nn.Module):
    """
    Applies a location-aware attention mechanism on the output features from the decoder.
    Location-aware attention proposed in "Attention-Based Models for Speech Recognition" paper.
    The location-aware attention mechanism is performing well in speech recognition tasks.
    Args:
        hidden_dim (int): The number of expected features in the output
        num_heads (int): The number of heads. (default: )
        conv_out_channel (int): The number of out channel in convolution
        smooting (bool): flag indication whether to use smoothing or not.
    Inputs: query, value, prev_attn, smoothing
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **prev_attn** (batch_size * num_heads, v_len): tensor containing previous timestep`s attention (alignment)
    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the feature from encoder outputs
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoder outputs.
    Reference:
        - **Attention-Based Models for Speech Recognition**: https://arxiv.org/abs/1506.07503
    """

    def __init__(self, hidden_dim, dim, conv_out_channel=10, smoothing=True):
        super(LocationAwareAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.dim = dim
        self.smoothing = smoothing
        self.loc_conv = nn.Conv1d(in_channels=1, out_channels=conv_out_channel, kernel_size=3, padding=1)
        self.loc_projection = nn.Linear(conv_out_channel, dim, bias=False)
        self.query_projection = nn.Linear(hidden_dim, dim, bias=False)
        self.value_projection = nn.Linear(hidden_dim, dim, bias=False)
        self.bias = nn.Parameter(torch.rand(dim).uniform_(-0.1, 0.1))
        self.score_projection = nn.Linear(dim, 1, bias=True)
        self.out_projection = nn.Linear(hidden_dim << 1, hidden_dim, bias=True)

    def forward(self, query, value, prev_attn, seq_len):
        batch_size, hidden_dim = query.size(0), query.size(2)
        residual = query

        # Initialize previous attention (alignment) to zeros
        if prev_attn is None:
            prev_attn = value.new_zeros(batch_size, seq_len)

        conv_feat = torch.transpose(self.loc_conv(prev_attn.unsqueeze(1)), 1, 2)
        score = self.score_projection(torch.tanh(
            self.query_projection(query.reshape(-1, hidden_dim)).view(batch_size, -1, self.dim)
            + self.value_projection(value.reshape(-1, hidden_dim)).view(batch_size, -1, self.dim)
            + self.loc_projection(conv_feat)
            + self.bias
        )).squeeze(dim=-1)

        if self.smoothing:
            score = torch.sigmoid(score)
            attn = torch.div(score, score.sum(dim=-1).unsqueeze(dim=-1))

        else:
            attn = function.softmax(score, dim=-1)

        context = torch.bmm(attn.unsqueeze(dim=1), value).squeeze(dim=1)

        # Get output
        combined = torch.cat([context, residual], dim=2)
        output = self.out_projection(combined.view(-1, self.hidden_dim << 1)).view(batch_size, -1, self.hidden_dim)
        return output, attn


class ScaledDotProductAttentions(nn.Module):
    """
    Scaled Dot-product Attention
    Args:
        dim (int): dimention of attention
    Inputs: query, value
        - **query** (batch_size, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch_size, v_len, hidden_dim): tensor containing features of the encoded input sequence.
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the alignment from the encoder outputs.
    Reference:
        - **Attention Is All You Need**: https://arxiv.org/abs/1706.03762
    """

    def __init__(self, dim):
        super(ScaledDotProductAttentions, self).__init__()
        self.dim = dim

    def forward(self, query, value):
        score = torch.bmm(query, value.transpose(1, 2)) / np.sqrt(self.dim)
        attn = function.softmax(score, dim=-1)
        context = torch.bmm(attn, value)
        return context, attn


class CustomizingAttention(nn.Module):
    r"""
    Customizing Attention
    Applies a multi-head + location-aware attention mechanism on the output features from the decoder.
    Multi-head attention proposed in "Attention Is All You Need" paper.
    Location-aware attention proposed in "Attention-Based Models for Speech Recognition" paper.
    I combined these two attention mechanisms as custom.
    Args:
        hidden_dim (int): The number of expected features in the output
        num_heads (int): The number of heads. (default: )
        conv_out_channel (int): The dimension of convolution
    Inputs: query, value, prev_attn
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **prev_attn** (batch_size * num_heads, v_len): tensor containing previous timestep`s alignment
    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch * num_heads, v_len): tensor containing the alignment from the encoder outputs.
    Reference:
        - **Attention Is All You Need**: https://arxiv.org/abs/1706.03762
        - **Attention-Based Models for Speech Recognition**: https://arxiv.org/abs/1506.07503
        - **State-Of-The-Art Speech Recognition with Sequence-to-Sequence Models**: https://arxiv.org/abs/1712.01769
    """

    def __init__(self, hidden_dim, num_heads=4, conv_out_channel=10):
        super(CustomizingAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim = int(hidden_dim / num_heads)
        self.scaled_dot = ScaledDotProductAttention(self.dim)
        self.conv1d = nn.Conv1d(1, conv_out_channel, kernel_size=3, padding=1)
        self.query_projection = nn.Linear(hidden_dim, self.dim * num_heads, bias=True)
        self.value_projection = nn.Linear(hidden_dim, self.dim * num_heads, bias=False)
        self.loc_projection = nn.Linear(conv_out_channel, self.dim, bias=False)
        self.bias = nn.Parameter(torch.rand(self.dim * num_heads).uniform_(-0.1, 0.1))
        self.out_projection = nn.Linear(hidden_dim << 1, hidden_dim, bias=True)

    def forward(self, query, value, prev_attn):
        batch_size, q_len, v_len = value.size(0), query.size(1), value.size(1)
        residual = query

        # Initialize previous attn (alignment) to zeros
        if prev_attn is None:
            prev_attn = value.new_zeros(batch_size * self.num_heads, v_len)

        loc_energy = self.get_loc_energy(prev_attn, batch_size, v_len)  # get location energy

        query = self.query_projection(query).view(batch_size, q_len, self.num_heads * self.dim)
        value = self.value_projection(value).view(batch_size, v_len, self.num_heads * self.dim) + loc_energy + self.bias

        query = query.view(batch_size, q_len, self.num_heads, self.dim).permute(2, 0, 1, 3)
        value = value.view(batch_size, v_len, self.num_heads, self.dim).permute(2, 0, 1, 3)
        query = query.contiguous().view(-1, q_len, self.dim)
        value = value.contiguous().view(-1, v_len, self.dim)

        context, attn = self.scaled_dot(query, value)
        attn = attn.squeeze()

        context = context.view(self.num_heads, batch_size, q_len, self.dim).permute(1, 2, 0, 3)
        context = context.contiguous().view(batch_size, q_len, -1)

        combined = torch.cat([context, residual], dim=2)
        output = torch.tanh(self.out_projection(combined.view(-1, self.hidden_dim << 1))).view(batch_size, -1,
                                                                                               self.hidden_dim)

        return output, attn

    def get_loc_energy(self, prev_attn, batch_size, v_len):
        conv_feat = self.conv1d(prev_attn.unsqueeze(1))
        conv_feat = conv_feat.view(batch_size, self.num_heads, -1, v_len).permute(0, 1, 3, 2)

        loc_energy = self.loc_projection(conv_feat).view(batch_size, self.num_heads, v_len, self.dim)
        loc_energy = loc_energy.permute(0, 2, 1, 3).reshape(batch_size, v_len, self.num_heads * self.dim)

        return loc_energy


class MultiHeadAttention(nn.Module):
    r"""
    Multi-Head Attention
    Applies a multi-headmechanism on the output features from the decoder.
    Multi-head attention proposed in "Attention Is All You Need" paper.
    Args:
        hidden_dim (int): The number of expected features in the output
        num_heads (int): The number of heads. (default: )
    Inputs: query, value, prev_align
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.
    Returns: output
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
    Reference:
        - **Attention Is All You Need**: https://arxiv.org/abs/1706.03762
        - **State-Of-The-Art Speech Recognition with Sequence-to-Sequence Models**: https://arxiv.org/abs/1712.01769
    """

    def __init__(self, hidden_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim = int(hidden_dim / num_heads)
        self.scaled_dot = ScaledDotProductAttentions(self.dim)
        self.query_projection = nn.Linear(hidden_dim, self.dim * num_heads)
        self.value_projection = nn.Linear(hidden_dim, self.dim * num_heads)
        self.out_projection = nn.Linear(hidden_dim << 1, hidden_dim, bias=True)

    def forward(self, query, value):
        batch_size = value.size(0)
        residual = query

        query = self.query_projection(query).view(batch_size, -1, self.num_heads, self.dim)
        value = self.value_projection(value).view(batch_size, -1, self.num_heads, self.dim)

        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.dim)
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.dim)

        context, _ = self.scaled_dot(query, value)
        context = context.view(self.num_heads, batch_size, -1, self.dim)

        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.dim)
        combined = torch.cat([context, residual], dim=2)

        output = torch.tanh(self.out_projection(combined.view(-1, self.hidden_dim << 1))).view(batch_size, -1,
                                                                                               self.hidden_dim)
        return output


class AdditiveAttention(nn.Module):
    """ Implementaion of Additive Attention (Bahdanau Attention) """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.score_projection = nn.Linear(hidden_dim, 1)
        self.out_projection = nn.Linear(hidden_dim << 1, hidden_dim, bias=True)

    def forward(self, query, value):
        score = self.score_projection(torch.tanh(
            self.value_projection(value)
            + self.query_projection(query.transpose(0, 1))
        )).squeeze(-1)
        attn = function.softmax(score, dim=1)
        context = torch.bmm(attn.unsqueeze(1), value)

        combined = torch.cat([context, query], dim=2)
        output = self.out_projection(combined.view(-1, self.hidden_dim << 1))
        return output


class ContentBasedAttention(nn.Module):
    """ Implementation of Content-based Attention """

    def __init__(self, hidden_dim, dim):
        super(ContentBasedAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.query_projection = nn.Linear(hidden_dim, dim, bias=False)
        self.value_projection = nn.Linear(hidden_dim, dim, bias=False)
        self.bias = nn.Parameter(torch.rand(dim).uniform_(-0.1, 0.1))
        self.score_projection = nn.Linear(dim, 1, bias=True)
        self.out_projection = nn.Linear(hidden_dim << 1, hidden_dim, bias=True)

    def forward(self, query, value):
        batch_size, hidden_dim = query.size(0), query.size(2)

        score = self.score_projection(self.tanh(
            self.query_projection(query.reshape(-1, hidden_dim)).view(batch_size, -1, self.attn_dim)
            + self.value_projection(value.reshape(-1, hidden_dim)).view(batch_size, -1, self.attn_dim)
            + self.bias
        )).squeeze(dim=-1)

        attn = self.softmax(score)
        context = torch.bmm(attn.unsqueeze(dim=1), value).squeeze(dim=1)

        combined = torch.cat([context, query], dim=2)
        output = self.out_projection(combined.view(-1, self.hidden_dim << 1))
        return output


class DotProductAttentions(nn.Module):
    """ Implementation of DotProduct Attention """

    def __init__(self, hidden_dim):
        super(DotProductAttentions, self).__init__()
        self.normalize = nn.LayerNorm(hidden_dim)
        self.out_projection = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, query, value):
        batch_size, hidden_dim, input_size = query.size(0), query.size(2), value.size(1)

        score = torch.bmm(query, value.transpose(1, 2))
        attn = function.softmax(score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        context = torch.bmm(attn, value)

        combined = torch.cat((context, query), dim=2)
        output = self.normalize(self.out_projection(combined.view(-1, 2 * hidden_dim))).view(batch_size, -1, hidden_dim)
        return output


class DotProductAttention(torch.nn.Module):
    """
    Compute the dot products of the query with all values and apply a softmax function to obtain the weights on the values
    """

    def __init__(self, hidden_dim):
        super(DotProductAttention, self).__init__()
        self.normalize = torch.nn.LayerNorm(hidden_dim)
        self.out_projection = torch.nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, query: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, hidden_dim, input_size = query.size(0), query.size(2), value.size(1)

        score = torch.bmm(query, value.transpose(1, 2))
        attn = torch.nn.functional.softmax(score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        context = torch.bmm(attn, value)

        return context, attn


class ScaledDotProductAttention(torch.nn.Module):
    """
    Scaled Dot-Product Attention

    Inputs: query, key, value, mask
        - query (batch, q_len, d_model): tensor containing projection vector for decoder.
        - key (batch, k_len, d_model): tensor containing projection vector for encoder.
        - value (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - mask (-): tensor containing indices to be masked
    """

    def __init__(self, dim: int):
        super().__init__()
        self.sqrt_dim = np.sqrt(dim)
        self.wq = torch.nn.Linear(in_features=dim, out_features=dim)
        self.wk = torch.nn.Linear(in_features=dim, out_features=dim)
        self.wv = torch.nn.Linear(in_features=dim, out_features=dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)

        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, query.size(1), -1)
            score.masked_fill_(mask, -float("Inf"))

        attn = torch.nn.functional.softmax(score, dim=-1)

        context = torch.bmm(attn, value)
        return context, attn


class DotProductAttention(torch.nn.Module):
    """
    Compute the dot products of the query with all values and apply a softmax function to obtain the weights on the values
    """

    def __init__(self, hidden_dim):
        super(DotProductAttention, self).__init__()
        self.normalize = torch.nn.LayerNorm(hidden_dim)
        self.out_projection = torch.nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, query: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, hidden_dim, input_size = query.size(0), query.size(2), value.size(1)

        score = torch.bmm(query, value.transpose(1, 2))
        attn = torch.nn.functional.softmax(score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        context = torch.bmm(attn, value)

        return context, attn
