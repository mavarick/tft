"""
Implementation of Temporal Fusion Transformers: https://arxiv.org/abs/1912.09363
"""

import math
import torch
import ipdb
import json
from torch import nn
import numpy as np
import torch.nn.functional as F
from layers.add_and_norm import AddAndNorm
from layers.gated_residual_network import GatedResidualNetwork
from layers.gated_linear_unit import GLU
from layers.linear_layer import LinearLayer
from layers.lstm_combine_and_mask import LSTMCombineAndMask
from layers.static_combine_and_mask import StaticCombineAndMask
from layers.time_distributed import TimeDistributed
from layers.interpretable_multi_head_attention import InterpretableMultiHeadAttention

from torch.utils.data import DataLoader,Dataset, Subset

USE_GELU = 1


class Transformer(nn.Module):
    def __init__(self, raw_params):
        super(Transformer, self).__init__()
        
        params = dict(raw_params)  # copy locally
        print(params)

        # Data parameters
        self.time_steps = int(params['total_time_steps'])
        self.input_size = int(params['input_size'])
        self.output_size = int(params['output_size'])
        self.category_counts = json.loads(str(params['category_counts']))
        self.n_multiprocessing_workers = int(params['multiprocessing_workers'])

        # Relevant indices for TFT
        self._input_obs_loc = json.loads(str(params['input_obs_loc']))
        self._static_input_loc = json.loads(str(params['static_input_loc']))
        self._known_regular_input_idx = json.loads(
            str(params['known_regular_inputs']))
        self._known_categorical_input_idx = json.loads(
            str(params['known_categorical_inputs']))

        self.column_definition = params['column_definition']

        # Network params
        self.quantiles = list(params['quantiles'])
        self.device = str(params['device'])
        self.hidden_layer_size = int(params['hidden_layer_size'])
        self.dropout_rate = float(params['dropout_rate'])
        self.max_gradient_norm = float(params['max_gradient_norm'])
        self.learning_rate = float(params['learning_rate'])
        self.minibatch_size = int(params['minibatch_size'])
        self.num_epochs = int(params['num_epochs'])
        self.early_stopping_patience = int(params['early_stopping_patience'])

        self.num_encoder_steps = int(params['num_encoder_steps'])
        self.num_stacks = int(params['stack_size'])
        self.num_heads = int(params['num_heads'])
        self.inputs_encoder = params['inputs_encoder']
        self.inputs_decoder = params['inputs_decoder']

        self.batch_first = True
        self.num_static = len(self._static_input_loc)
        #self.num_inputs = len(self._known_regular_input_idx) + len(self._known_categorical_input_idx) + self.output_size
        #self.num_inputs = len(self._known_regular_input_idx) + self.output_size
        #self.num_inputs_decoder = len(self._known_regular_input_idx)
        #self.num_inputs = len(self._known_regular_input_idx) + len(self._known_categorical_input_idx) + self.output_size
        #self.num_inputs_decoder = len(self._known_regular_input_idx) + len(self._known_categorical_input_idx)
        self.num_inputs = len(self.inputs_encoder) - self.num_static
        self.num_inputs_decoder = len(self.inputs_decoder) - self.num_static

        print("_known_regular_input_idx:{}, _known_categorical_input_idx:{}, num_static: {}, num_inputs:{}".format(
            self._known_regular_input_idx, self._known_categorical_input_idx, self.num_static, self.num_inputs))
        # Serialisation options
        # self._temp_folder = os.path.join(params['model_folder'], 'tmp')
        # self.reset_temp_folder()

        # Extra components to store Tensorflow nodes for attention computations
        self._input_placeholder = None
        self._attention_components = None
        self._prediction_parts = None

        # print('*** params ***')
        # for k in params:
        #   print('# {} = {}'.format(k, params[k]))
        
        #######
        time_steps = self.time_steps
        num_categorical_variables = len(self.category_counts)
        num_regular_variables = self.input_size - num_categorical_variables

        embedding_sizes = [
          self.hidden_layer_size for i, size in enumerate(self.category_counts)
        ]

        print("num_categorical_variables")
        print(num_categorical_variables)
        self.embeddings = nn.ModuleList()
        for i in range(num_categorical_variables):
            embedding = nn.Embedding(self.category_counts[i], embedding_sizes[i])
            self.embeddings.append(embedding)

        self.static_input_layer = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        self.time_varying_embedding_layer = LinearLayer(input_size=1, size=self.hidden_layer_size,
                                                        use_time_distributed=True, batch_first=self.batch_first)

        ## transformer的参数
        # 输入历史特征天级别特征大小, 注意static是否加入
        self.model_enc_dim = int(params['model_enc_dim'])
        # 输入未来特征天级别特征大小, 注意static是否加入
        self.model_dec_dim = int(params['model_dec_dim'])
        # encoder和decoder的天级别变化特征大小
        self.model_dim = int(params['model_dim'])

        self.num_layers = int(params.get('num_layers', 2))
        self.num_heads = int(params.get('num_heads', 4))
        self.use_conv = int(params.get("use_conv", 1))  # 是否使用卷积来进行qkv变换
        self.ffn_dim = int(params.get("ffn_dim", 64))  # attention之后的feedforward的隐藏层节点个数
        self.dropout = float(params.get("dropout", 0.1)) #

        self.attn_log = int(params.get("attn_log", 1))   # atten在softmax之前是否要log
        self.layer_norm = int(params.get("layer_norm", 1))  #

        self.encoder = Encoder(self.num_layers, self.model_enc_dim, self.model_dim,
                               self.num_heads, self.ffn_dim, self.dropout, use_conv=self.use_conv)
        self.decoder = Decoder(self.num_layers, self.model_dec_dim, self.model_dim,
                               self.num_heads, self.ffn_dim, self.dropout, use_conv=False)

        self.linear = nn.Linear(self.model_dim, 1, bias=False)

    def get_decoder_mask(self, self_attn_inputs):
        """Returns causal mask to apply for self-attention layer.

        Args:
          self_attn_inputs: Inputs to self attention layer to determine mask shape
        """
        len_s = self_attn_inputs.shape[1] # 192
        bs = self_attn_inputs.shape[:1][0] # [64]
        # create batch_size identity matrices
        mask = torch.cumsum(torch.eye(len_s).reshape((1, len_s, len_s)).repeat(bs, 1, 1), 1)
        return mask

    def get_tft_embeddings(self, all_inputs):
        time_steps = self.time_steps
        
        num_categorical_variables = len(self.category_counts)
        num_regular_variables = self.input_size - num_categorical_variables
        # 5 - 1 = 4

        embedding_sizes = [
          self.hidden_layer_size for i, size in enumerate(self.category_counts)
        ]
        #embedding_sizes = [160]

        ## 这个地方就要求，recular_inputs全部放在category之前, TODO
        regular_inputs, categorical_inputs \
            = all_inputs[:, :, :num_regular_variables], \
              all_inputs[:, :, num_regular_variables:]

        regular_inputs = regular_inputs.float()

        embedded_inputs = [
            self.embeddings[i](categorical_inputs[:,:, i].long())
            for i in range(num_categorical_variables)
        ]

        #print("num_regular_variables:{}, regular_inputs:{}, categorical_inputs:{}".format(
        #    num_regular_variables, regular_inputs.shape, categorical_inputs.shape))

        # Static inputs
        if self._static_input_loc:
            static_inputs = [self.static_input_layer(
                regular_inputs[:, 0, i:i + 1]) for i in range(num_regular_variables)
                              if i in self._static_input_loc] \
                + [embedded_inputs[i][:, 0, :]
                    for i in range(num_categorical_variables)
                    if i + num_regular_variables in self._static_input_loc]
            static_inputs = torch.stack(static_inputs, dim=1)
        else:
            static_inputs = None

        # Targets
        ## 这个地方问题是为什么要加一个time_varying_embedding_layer ??
        obs_inputs = torch.stack([
            self.time_varying_embedding_layer(regular_inputs[Ellipsis, i:i + 1].float())
            for i in self._input_obs_loc
        ], dim=-1)

        # Observed (a prioir unknown) inputs
        wired_embeddings = []
        for i in range(num_categorical_variables):
            if i not in self._known_categorical_input_idx and i not in self._input_obs_loc:
                e = self.embeddings[i](categorical_inputs[:, :, i])
                wired_embeddings.append(e)

        unknown_inputs = []
        for i in range(regular_inputs.shape[-1]):
            if i not in self._known_regular_input_idx and i not in self._input_obs_loc:
                e = self.time_varying_embedding_layer(regular_inputs[Ellipsis, i:i + 1])
                unknown_inputs.append(e)

        if unknown_inputs + wired_embeddings:
            unknown_inputs = torch.stack(unknown_inputs + wired_embeddings, dim=-1)
        else:
            unknown_inputs = None

        # A priori known inputs
        known_regular_inputs = [
            self.time_varying_embedding_layer(regular_inputs[Ellipsis, i:i + 1].float())
            for i in self._known_regular_input_idx
            if i not in self._static_input_loc
        ]
        known_categorical_inputs = [
            embedded_inputs[i]
            for i in self._known_categorical_input_idx
            if i + num_regular_variables not in self._static_input_loc
        ]

        known_combined_layer = torch.stack(known_regular_inputs + known_categorical_inputs, dim=-1)

        return unknown_inputs, known_combined_layer, obs_inputs, static_inputs

    def forward(self, x):
        # Size definitions.
        time_steps = self.time_steps
        combined_input_size = self.input_size
        encoder_steps = self.num_encoder_steps
        all_inputs = x.to(self.device)

        unknown_inputs, known_combined_layer, obs_inputs, static_inputs \
            = self.get_tft_embeddings(all_inputs)
        # 这个地方返回分别对应：未来不知道，未来知道, labels, 静态特征
        print("all_inputs:{}, unknown_inputs:{}, known_combined_layer:{}, obs_inputs:{}, static_inputs:{}".format(
           all_inputs.shape,
           None if unknown_inputs is None else unknown_inputs.shape, known_combined_layer.shape, obs_inputs.shape, static_inputs.shape
        ))

        # Isolate known and observed historical inputs.
        #print("unknown_inputs:", unknown_inputs)

        if unknown_inputs is not None:
            historical_inputs = torch.cat([
                #static_inputs[:, :encoder_steps, :],
                unknown_inputs[:, :encoder_steps, :],
                known_combined_layer[:, :encoder_steps, :],
                obs_inputs[:, :encoder_steps, :]
            ], dim=-1)
        else:
            historical_inputs = torch.cat([
                  #static_inputs[:, :encoder_steps, :],
                  known_combined_layer[:, :encoder_steps, :],
                  obs_inputs[:, :encoder_steps, :]
              ], dim=-1)

        # static怎么加入的问题，TODO
        # Isolate only known future inputs.
        future_inputs = known_combined_layer[:, encoder_steps:, :]

        #print("historical_inputs:{}, future_inputs:{}".format(historical_inputs.shape, future_inputs.shape))
        enc_output, enc_self_attn = self.encoder(historical_inputs)

        context_attn_mask = None
        output, dec_self_attn, ctx_attn = self.decoder(
            future_inputs, enc_output, context_attn_mask)

        output = self.linear(output)
        return output, enc_self_attn, dec_self_attn, ctx_attn


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        # self.softmax = nn.Softmax(dim=2)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None, attn_log=1):
        """前向传播.

        Args:
        	q: Queries张量，形状为[B, L_q, D_q]
        	k: Keys张量，形状为[B, L_k, D_k]
        	v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
        	scale: 缩放因子，一个浮点标量
        	attn_mask: Masking张量，形状为[B, L_q, L_k]

        Returns:
        	上下文张量和attetention张量
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask:
            # 给需要mask的地方设置一个负无穷
            attention = attention.masked_fill_(attn_mask, -np.inf)
        # 计算softmax, enc_attention几乎所有值都为0，为了防止这样子，这个地方特意
        if attn_log:
            attention = attention.masked_fill_(attention < 0.1, 0.1)
            attention = torch.log(attention)
            # attention = torch.sign(attention) * torch.sqrt(torch.abs(attention))
            # pass

        attention = self.softmax(attention)
        if torch.sum(torch.isnan(attention.cpu())) > 0.1:
            print(q)
            print(k)
            raise Exception("attention is null")
        # 添加dropout
        attention = self.dropout(attention)
        # 和V做点积
        context = torch.bmm(attention, v)
        return context, attention


class ConvCasual1D(nn.Module):
    """用cnn的方式来计算kqv(因果卷积)
    """

    def __init__(self, input_size, output_size=1, kernel_size=3):
        super(ConvCasual1D, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size

        # padding = 0
        # 二维conv：https://blog.csdn.net/m0_37586991/article/details/87855342
        # self.conv2 = nn.Conv2d()
        # 卷积
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=output_size, kernel_size=self.kernel_size,
                               padding=(self.kernel_size - 1), padding_mode="circular")
        # self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=output_size, kernel_size=self.kernel_size, padding=2*(self.kernel_size-1), padding_mode="same")
        # shape: bs, output_size, new_len

        # self.active_func = nn.SELU()
        self.active_func = nn.GELU()
        # self.active_func = nn.ReLU()

        # output_size

    def forward(self, input):
        ## input.shape: bs, nday, d_model
        ## torch.nn.Conv1d
        input = input.permute(0, 2, 1)  # bs, d_model, nday
        # print(input.shape)
        output = self.conv1(input)
        # 因果卷积, bs, output_size, new_len
        output = output[:, :, :-(self.kernel_size - 1)]
        output = output.permute(0, 2, 1)

        # NEW add gelu
        return self.active_func(output)


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, dropout=0.0, use_conv=True):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        if not use_conv:
            self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
            self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
            self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)
        else:
            output_size = self.dim_per_head * num_heads
            self.linear_k = ConvCasual1D(input_size=model_dim, output_size=output_size, kernel_size=3)
            self.linear_v = ConvCasual1D(input_size=model_dim, output_size=output_size, kernel_size=3)
            self.linear_q = ConvCasual1D(input_size=model_dim, output_size=output_size, kernel_size=3)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        # multi-head attention之后需要做layer norm
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None, attn_log=1, layer_norm=1):
        # 残差连接
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.contiguous().view(batch_size * num_heads, -1, dim_per_head)
        value = value.contiguous().view(batch_size * num_heads, -1, dim_per_head)
        query = query.contiguous().view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask, attn_log=attn_log)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        if layer_norm:
            output = self.layer_norm(residual + output)
        else:
            output = residual + output

        return output, attention


class LayerNorm(nn.Module):
    """实现LayerNorm。其实PyTorch已经实现啦，见nn.LayerNorm。
    这个layernorm是否有必要？
    """

    def __init__(self, features, epsilon=1e-6):
        """ feature, 模型的维度
        """
        super(LayerNorm, self).__init__()
        # alpha
        self.gamma = nn.Parameter(torch.ones(features))
        # beta
        self.beta = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x):
        """
        """
        # 根据公式进行归一化
        # 在X的最后一个维度求均值，最后一个维度就是模型的维度
        mean = x.mean(-1, keepdim=True)
        # 在X的最后一个维度求方差，最后一个维度就是模型的维度
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta


# def padding_mask(seq_k, seq_q):
#     # seq_k和seq_q的形状都是[B,L]
#     len_q = seq_q.size(1)
#     # `PAD` is 0
#     pad_mask = seq_k.eq(0)
#     pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
#     return pad_mask

# def sequence_mask(seq):
#     batch_size, seq_len = seq.size()
#     mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),
#                     diagonal=1)
#     mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
#     return mask

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len):
        """初始化。

            d_model: 一个标量。模型的维度，论文默认是512
            max_seq_len: 一个标量。文本序列的最大长度
        """
        super(PositionalEncoding, self).__init__()

        # 根据论文给的公式，构造出PE矩阵
        position_encoding = np.array([
            [pos / np.pow(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])
        # 偶数列使用sin，奇数列使用cos
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        # 在PE矩阵的第一行，加上一行全是0的向量，代表这`PAD`的positional encoding
        # 在word embedding中也经常会加上`UNK`，代表位置单词的word embedding，两者十分类似
        # 那么为什么需要这个额外的PAD的编码呢？很简单，因为文本序列的长度不一，我们需要对齐，
        # 短的序列我们使用0在结尾补全，我们也需要这些补全位置的编码，也就是`PAD`对应的位置编码
        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, position_encoding))

        # 嵌入操作，+1是因为增加了`PAD`这个补全位置的编码，
        # Word embedding中如果词典增加`UNK`，我们也需要+1。看吧，两者十分相似
        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)

    def forward(self, input_len):
        """神经网络的前向传播。

        Args:
          input_len: 一个张量，形状为[BATCH_SIZE, 1]。每一个张量的值代表这一批文本序列中对应的长度。

        Returns:
          返回这一批序列的位置编码，进行了对齐。
        """

        # 找出这一批序列的最大长度
        max_len = torch.max(input_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        # 对每一个序列的位置进行对齐，在原序列位置的后面补上0
        # 这里range从1开始也是因为要避开PAD(0)的位置
        input_pos = tensor(
            [list(range(1, len + 1)) + [0] * (max_len - len) for len in input_len])
        return self.position_encoding(input_pos)


class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        # self.gelu = nn.GELU()
        # self.active_func = nn.ReLU()
        # self.active_func = nn.SELU()
        self.active_func = nn.GELU()

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x, layer_norm=1):
        output = x.transpose(1, 2)

        if USE_GELU:
            output = self.w2(self.active_func(self.w1(output)))
        else:
            output = self.w2(F.relu(self.w1(output)))

        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        if layer_norm:
            output = self.layer_norm(x + output)
        else:
            output = x + output
        return output


class EncoderLayer(nn.Module):
    """Encoder的一层。"""

    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2018, dropout=0.0, use_conv=True):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout, use_conv=use_conv)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None, attn_log=1, layer_norm=1):
        # self attention
        context, attention = self.attention(inputs, inputs, inputs, attn_mask, attn_log=attn_log, layer_norm=layer_norm)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention


class Encoder(nn.Module):
    """多层EncoderLayer组成Encoder。"""

    def __init__(self,
                 # vocab_size,
                 # max_seq_len,
                 num_layers=6,
                 model_enc_dim=512,
                 model_dim=512,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.0,
                 use_conv=True,
                 ):
        super(Encoder, self).__init__()

        self.encoder_normdim_layer = nn.Linear(model_enc_dim, model_dim, bias=True)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, num_heads, ffn_dim, dropout, use_conv=use_conv) for _ in
             range(num_layers)])

        # self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
        #self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

    def forward(self, inputs, attn_log=1, layer_norm=1):
        # output = self.seq_embedding(inputs)
        # output += self.pos_embedding(inputs_len)

        # self_attention_mask = padding_mask(inputs, inputs)

        output = self.encoder_normdim_layer(inputs)

        attentions = []
        for encoder in self.encoder_layers:
            # output, attention = encoder(inputs, self_attention_mask)
            output, attention = encoder(output, attn_log=attn_log, layer_norm=layer_norm)
            attentions.append(attention)

        return output, attentions


class DecoderLayer(nn.Module):

    def __init__(self, model_dim, num_heads=8, ffn_dim=2048, dropout=0.0, use_conv=False):
        super(DecoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout, use_conv=use_conv)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self,
                dec_inputs,
                enc_outputs,
                self_attn_mask=None,
                context_attn_mask=None,
                attn_log=1,
                layer_norm=1
                ):
        # self attention, all inputs are decoder inputs
        dec_output, self_attention = self.attention(
            dec_inputs, dec_inputs, dec_inputs, self_attn_mask, attn_log=attn_log, layer_norm=layer_norm)

        # context attention
        # query is decoder's outputs, key and value are encoder's inputs
        dec_output, context_attention = self.attention(
            enc_outputs, enc_outputs, dec_output, context_attn_mask, attn_log=attn_log, layer_norm=layer_norm)

        # decoder's output, or context
        dec_output = self.feed_forward(dec_output)

        return dec_output, self_attention, context_attention


class Decoder(nn.Module):

    def __init__(self,
                 # vocab_size,
                 # max_seq_len,
                 num_layers=6,
                 model_dec_dim=28,
                 model_dim=64,
                 num_heads=6,
                 ffn_dim=64,
                 dropout=0.0,
                 use_conv=False,
                 ):
        super(Decoder, self).__init__()

        self.num_layers = num_layers

        self.decoder_normdim_layer = nn.Linear(model_dec_dim, model_dim, bias=False)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(model_dim, num_heads, ffn_dim, dropout, use_conv=use_conv) for _ in
             range(num_layers)])

        # self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
        # self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

    def forward(self, inputs, enc_output, context_attn_mask=None, attn_log=1, layer_norm=1):
        # output = self.seq_embedding(inputs)
        # output += self.pos_embedding(inputs_len)

        # self_attention_padding_mask = padding_mask(inputs, inputs)
        # seq_mask = sequence_mask(inputs)
        # self_attn_mask = torch.gt((self_attention_padding_mask + seq_mask), 0)

        self_attn_mask = None
        context_attn_mask = None

        self_attentions = []
        context_attentions = []
        output = self.decoder_normdim_layer(inputs)
        for decoder in self.decoder_layers:
            output, self_attn, context_attn = decoder(
                output, enc_output, self_attn_mask, context_attn_mask, attn_log=attn_log, layer_norm=layer_norm)
            self_attentions.append(self_attn)
            context_attentions.append(context_attn)

        return output, self_attentions, context_attentions


from torch.nn import ModuleDict


class SparseFeatureDictNet(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, his_features_infos, pred_features_infos, use_onehot=True, use_onehot_freeze=True, sparse_di=16):
        """ 这个地方把所有的离散型的变量变换成长度一样的embedding，
        TODO: 可以在配置文件中设置各个离散变量的embedding的长度.

        如果use_onehot=True,那么使用onehot的值
        """
        super(SparseFeatureDictNet, self).__init__()
        self.his_features_infos = his_features_infos
        self.pred_features_infos = pred_features_infos
        self.sparse_di = sparse_di
        self.use_onehot = use_onehot
        self.use_onehot_freeze = use_onehot_freeze

        feature_class_map = self._parse_sparse_features()

        self.embedding_net_dict = ModuleDict()
        for col_name, n_class in feature_class_map.items():
            if use_onehot:
                self.embedding_net_dict[col_name] = nn.Embedding.from_pretrained(torch.eye(n_class),
                                                                                 freeze=use_onehot_freeze)
            else:
                self.embedding_net_dict[col_name] = nn.Embedding(n_class, self.sparse_di)

    def _parse_sparse_features(self):
        feature_class_num_map = {}
        for f in self.his_features_infos + self.pred_features_infos:
            col_idx, col_name, is_sparse, n_class = f
            if n_class > 0:
                feature_class_num_map[col_name] = n_class
        return feature_class_num_map

    def forward(self, sparse_input, sparse_column):
        """ 输入一个sparse features，然后返回这个sparse_input的embeding
        sparse_input:: shape: (batch_size, nday)
        sparse_column:: str

        return: (batch_size, nday, embed_size)
        """
        assert sparse_column in self.embedding_net_dict, "输入的离散变量列名必须存在:[{}]".format(sparse_column)
        return self.embedding_net_dict[sparse_column](sparse_input)
