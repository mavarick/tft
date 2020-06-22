"""
Implementation of Temporal Fusion Transformers: https://arxiv.org/abs/1912.09363
"""


from torch import nn
import math
import torch
import ipdb
import json

class QuantileLoss(nn.Module):
    ## From: https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629

    def __init__(self, quantiles):
        ##takes a list of quantiles
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(
                torch.max(
                   (q-1) * errors, 
                   q * errors
            ).unsqueeze(1))
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss

class TimeDistributed(nn.Module):
    ## Takes any module and stacks the time dimension with the batch dimenison of inputs before apply the module
    ## From: https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class AddAndNorm(nn.Module):
    def __init__(self, hidden_layer_size):
        super(AddAndNorm, self).__init__()

        self.normalize = nn.LayerNorm(hidden_layer_size)

    def forward(self, x1, x2):
        x = torch.add(x1, x2)
        return self.normalize(x)

class LinearLayer(nn.Module):
    def __init__(self,
                input_size,
                size,
                use_time_distributed=True,
                batch_first=False):
        super(LinearLayer, self).__init__()

        self.use_time_distributed=use_time_distributed
        self.input_size=input_size
        self.size=size
        if use_time_distributed:
            self.layer = TimeDistributed(nn.Linear(input_size, size), batch_first=batch_first)
        else:
            self.layer = nn.Linear(input_size, size)
      
    def forward(self, x):
        return self.layer(x)

class GLU(nn.Module):
    #Gated Linear Unit
    def __init__(self, 
                input_size,
                hidden_layer_size,
                dropout_rate=None,
                use_time_distributed=True,
                batch_first=False
                ):
        super(GLU, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.use_time_distributed = use_time_distributed

        if dropout_rate is not None:
            self.dropout = nn.Dropout(self.dropout_rate)
        
        self.activation_layer = LinearLayer(input_size, hidden_layer_size, use_time_distributed, batch_first)
        self.gated_layer = LinearLayer(input_size, hidden_layer_size, use_time_distributed, batch_first)

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        if self.dropout_rate is not None:
            x = self.dropout(x)
        
        activation = self.activation_layer(x)
        gated = self.sigmoid(self.gated_layer(x))
        
        return torch.mul(activation, gated), gated


class GatedResidualNetwork(nn.Module):
    def __init__(self, 
                input_size,
                hidden_layer_size,
                output_size=None,
                dropout_rate=None,
                use_time_distributed=True,
                return_gate=False,
                batch_first=False
                ):

        super(GatedResidualNetwork, self).__init__()
        if output_size is None:
            output = hidden_layer_size
        else:
            output = output_size
        
        self.output = output
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_size = hidden_layer_size
        self.return_gate = return_gate

        self.linear_layer = LinearLayer(input_size, output, use_time_distributed, batch_first)

        self.hidden_linear_layer1 = LinearLayer(input_size, hidden_layer_size, use_time_distributed, batch_first)
        self.hidden_context_layer = LinearLayer(hidden_layer_size, hidden_layer_size, use_time_distributed, batch_first)
        self.hidden_linear_layer2 = LinearLayer(hidden_layer_size, hidden_layer_size, use_time_distributed, batch_first)

        self.elu1 = nn.ELU()
        self.glu = GLU(hidden_layer_size, output, dropout_rate, use_time_distributed, batch_first)
        self.add_and_norm = AddAndNorm(hidden_layer_size=output)

    def forward(self, x, context=None):
        # Setup skip connection
        if self.output_size is None:
            skip = x
        else:
            skip = self.linear_layer(x)

        # Apply feedforward network
        hidden = self.hidden_linear_layer1(x)
        if context is not None:
            hidden = hidden + self.hidden_context_layer(context)
        hidden = self.elu1(hidden)
        hidden = self.hidden_linear_layer2(hidden)

        gating_layer, gate = self.glu(hidden)
        if self.return_gate:
            return self.add_and_norm(skip, gating_layer), gate
        else:
            return self.add_and_norm(skip, gating_layer)

# class LambdaLayer(nn.Module):
#     # https://discuss.pytorch.org/t/how-to-implement-keras-layers-core-lambda-in-pytorch/5903/2
#     def __init__(self, lambd):
#         super(LambdaLayer, self).__init__()
#         self.lambd = lambd
#     def forward(self, x):
#         return self.lambd(x)

# class ScaledDotProductAttention():
#     """Defines scaled dot product attention layer.

#     Attributes:
#       dropout: Dropout rate to use
#     """

#     def __init__(self, attn_dropout=0.0):
#       super(ScaledDotProductAttention, self).__init__()
#       self.dropout = nn.Dropout(attn_dropout)
#       self.activation = nn.Softmax()
#       self.lambda_layer1 = 

#     def forward(self, q, k, v, mask):
#         """Applies scaled dot product attention.

#         Args:
#           q: Queries
#           k: Keys
#           v: Values
#           mask: Masking if required -- sets softmax to very large value

#         Returns:
#           Tuple of (layer outputs, attention weights)
#         """
#         temper = torch.sqrt(torch.tensor(k.shape[-], dtype=torch.float32))
#         # attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / temper)(
#         #     [q, k])  # shape=(batch, q, k)
#         if mask is not None:
#             # mmask = Lambda(lambda x: (-1e+9) * (1. - K.cast(x, 'float32')))(
#             #     mask)  # setting to infinity
#             attn = torch.add(attn, mmask)
#         attn = self.activation(attn)
#         attn = self.dropout(attn)
#         # output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
#         return output, attn




# class InterpretableMultiHeadAttention():
#     """Defines interpretable multi-head attention layer.

#     Attributes:
#       n_head: Number of heads
#       d_k: Key/query dimensionality per head
#       d_v: Value dimensionality
#       dropout: Dropout rate to apply
#       qs_layers: List of queries across heads
#       ks_layers: List of keys across heads
#       vs_layers: List of values across heads
#       attention: Scaled dot product attention layer
#       w_o: Output weight matrix to project internal state to the original TFT
#         state size
#     """

#     def __init__(self, n_head, d_model, dropout):
#         """Initialises layer.

#         Args:
#           n_head: Number of heads
#           d_model: TFT state dimensionality
#           dropout: Dropout discard rate
#         """

#       self.n_head = n_head
#       self.d_k = self.d_v = d_k = d_v = d_model // n_head
#       self.dropout = dropout

#       self.qs_layers = []
#       self.ks_layers = []
#       self.vs_layers = []

#       # Use same value layer to facilitate interp
#       vs_layer = Dense(d_v, use_bias=False)
#       qs_layer = Dense(d_k, use_bias=False)
#       ks_layer = Dense(d_k, use_bias=False)

#       for _ in range(n_head):
#         self.qs_layers.append(qs_layer)
#         self.ks_layers.append(ks_layer)
#         self.vs_layers.append(vs_layer)  # use same vs_layer

#       self.attention = ScaledDotProductAttention()
#       self.w_o = Dense(d_model, use_bias=False)

#     def __call__(self, q, k, v, mask=None):
#       """Applies interpretable multihead attention.

#       Using T to denote the number of time steps fed into the transformer.

#       Args:
#         q: Query tensor of shape=(?, T, d_model)
#         k: Key of shape=(?, T, d_model)
#         v: Values of shape=(?, T, d_model)
#         mask: Masking if required with shape=(?, T, T)

#       Returns:
#         Tuple of (layer outputs, attention weights)
#       """
#       n_head = self.n_head

#       heads = []
#       attns = []
#       for i in range(n_head):
#         qs = self.qs_layers[i](q)
#         ks = self.ks_layers[i](k)
#         vs = self.vs_layers[i](v)
#         head, attn = self.attention(qs, ks, vs, mask)

#         head_dropout = Dropout(self.dropout)(head)
#         heads.append(head_dropout)
#         attns.append(attn)
#       head = K.stack(heads) if n_head > 1 else heads[0]
#       attn = K.stack(attns)

#       outputs = K.mean(head, axis=0) if n_head > 1 else head
#       outputs = self.w_o(outputs)
#       outputs = Dropout(self.dropout)(outputs)  # output dropout

#       return outputs, attn

class LSTMCombineAndMask(nn.Module):
    def __init__(self, input_size, num_inputs, hidden_layer_size, dropout_rate, additional_context=None, use_time_distributed=False, batch_first=True):
        super(LSTMCombineAndMask, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size
        self.num_inputs = num_inputs
        self.dropout_rate = dropout_rate
        self.additional_context= additional_context

        if self.additional_context is not None:
            self.flattened_grn = GatedResidualNetwork(self.num_inputs*self.hidden_layer_size, self.hidden_layer_size, self.num_inputs, self.dropout_rate, use_time_distributed=use_time_distributed, return_gate=True, batch_first=batch_first)
        else:
            self.flattened_grn = GatedResidualNetwork(self.num_inputs*self.hidden_layer_size, self.hidden_layer_size, self.num_inputs, self.dropout_rate, use_time_distributed=use_time_distributed, return_gate=True, batch_first=batch_first)


        self.single_variable_grns = nn.ModuleList()
        for i in range(self.num_inputs):
            self.single_variable_grns.append(GatedResidualNetwork(self.hidden_layer_size, self.hidden_layer_size, None, self.dropout_rate, use_time_distributed=use_time_distributed, return_gate=False, batch_first=batch_first))

        self.softmax = nn.Softmax(dim=2)

    def forward(self, embedding, additional_context=None):
        # Add temporal features
        _, time_steps, embedding_dim, num_inputs = list(embedding.shape)
                
        flattened_embedding = torch.reshape(embedding,
                      [-1, time_steps, embedding_dim * num_inputs])

        expanded_static_context = additional_context.unsqueeze(1)

        if additional_context is not None:
            sparse_weights, static_gate = self.flattened_grn(flattened_embedding, expanded_static_context)
        else:
            sparse_weights = self.flattened_grn(flattened_embedding)

        sparse_weights = self.softmax(sparse_weights).unsqueeze(2)

        trans_emb_list = []
        for i in range(self.num_inputs):
            ##select slice of embedding belonging to a single input
            trans_emb_list.append(
              self.single_variable_grns[i](embedding[Ellipsis,i])
            )

        transformed_embedding = torch.stack(trans_emb_list, dim=-1)
        
        combined = transformed_embedding*sparse_weights
        
        temporal_ctx = combined.sum(dim=-1)

        return temporal_ctx, sparse_weights, static_gate

class StaticCombineAndMask(nn.Module):
    def __init__(self, input_size, num_static, hidden_layer_size, dropout_rate, additional_context=None, use_time_distributed=False, batch_first=True):
        super(StaticCombineAndMask, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.input_size =input_size
        self.num_static = num_static
        self.dropout_rate = dropout_rate
        self.additional_context = additional_context

        if self.additional_context is not None:
            self.flattened_grn = GatedResidualNetwork(self.num_static*self.hidden_layer_size, self.hidden_layer_size, self.num_static, self.dropout_rate, use_time_distributed=False, return_gate=False, batch_first=batch_first)
        else:
            self.flattened_grn = GatedResidualNetwork(self.num_static*self.hidden_layer_size, self.hidden_layer_size, self.num_static, self.dropout_rate, use_time_distributed=False, return_gate=False, batch_first=batch_first)


        self.single_variable_grns = nn.ModuleList()
        for i in range(self.num_static):
            self.single_variable_grns.append(GatedResidualNetwork(self.hidden_layer_size, self.hidden_layer_size, None, self.dropout_rate, use_time_distributed=False, return_gate=False, batch_first=batch_first))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, embedding, additional_context=None):
        # Add temporal features
        _, num_static, _ = list(embedding.shape)
        flattened_embedding = torch.flatten(embedding, start_dim=1)
        if additional_context is not None:
            sparse_weights = self.flattened_grn(flattened_embedding, additional_context)
        else:
            sparse_weights = self.flattened_grn(flattened_embedding)

        sparse_weights = self.softmax(sparse_weights).unsqueeze(2)

        trans_emb_list = []
        for i in range(self.num_static):
            ##select slice of embedding belonging to a single input
            trans_emb_list.append(
              self.single_variable_grns[i](torch.flatten(embedding[:, i:i + 1, :], start_dim=1))
            )

        transformed_embedding = torch.stack(trans_emb_list, dim=1)

        combined = transformed_embedding*sparse_weights
        
        static_vec = combined.sum(dim=1)

        return static_vec, sparse_weights

class TFT(nn.Module):
    def __init__(self, raw_params):
        super(TFT, self).__init__()
        
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
        self.quantiles = [0.1, 0.5, 0.9]
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
        self.batch_first = True
        self.num_static = len(self._static_input_loc)
        self.num_inputs = len(self._known_regular_input_idx) + self.output_size
        self.num_inputs_decoder = len(self._known_regular_input_idx)

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
        self.time_varying_embedding_layer = LinearLayer(input_size=1, size=self.hidden_layer_size, use_time_distributed=True, batch_first=self.batch_first)

        self.static_combine_and_mask = StaticCombineAndMask(
                input_size=self.input_size,
                num_static=self.num_static,
                hidden_layer_size=self.hidden_layer_size,
                dropout_rate=self.dropout_rate,
                additional_context=None,
                use_time_distributed=False,
                batch_first=self.batch_first)
        self.static_context_variable_selection_grn = GatedResidualNetwork(
                input_size=self.hidden_layer_size,
                hidden_layer_size=self.hidden_layer_size,
                output_size=None,
                dropout_rate=self.dropout_rate,
                use_time_distributed=False,
                return_gate=False,
                batch_first=self.batch_first)
        self.static_context_enrichment_grn = GatedResidualNetwork(
                input_size=self.hidden_layer_size,
                hidden_layer_size=self.hidden_layer_size,
                output_size=None,
                dropout_rate=self.dropout_rate,
                use_time_distributed=False,
                return_gate=False,
                batch_first=self.batch_first)
        self.static_context_state_h_grn = GatedResidualNetwork(
                input_size=self.hidden_layer_size,
                hidden_layer_size=self.hidden_layer_size,
                output_size=None,
                dropout_rate=self.dropout_rate,
                use_time_distributed=False,
                return_gate=False,
                batch_first=self.batch_first)
        self.static_context_state_c_grn = GatedResidualNetwork(
                input_size=self.hidden_layer_size,
                hidden_layer_size=self.hidden_layer_size,
                output_size=None,
                dropout_rate=self.dropout_rate,
                use_time_distributed=False,
                return_gate=False,
                batch_first=self.batch_first)
        self.historical_lstm_combine_and_mask = LSTMCombineAndMask(
                input_size=self.num_encoder_steps,
                num_inputs=self.num_inputs,
                hidden_layer_size=self.hidden_layer_size,
                dropout_rate=self.dropout_rate,
                additional_context=True,
                use_time_distributed=True,
                batch_first=self.batch_first)
        self.future_lstm_combine_and_mask = LSTMCombineAndMask(
                input_size=self.num_encoder_steps,
                num_inputs=self.num_inputs_decoder,
                hidden_layer_size=self.hidden_layer_size,
                dropout_rate=self.dropout_rate,
                additional_context=True,
                use_time_distributed=True,
                batch_first=self.batch_first)

        self.lstm_encoder = nn.LSTM(input_size=self.hidden_layer_size, hidden_size=self.hidden_layer_size, batch_first=self.batch_first)
        self.lstm_decoder = nn.LSTM(input_size=self.hidden_layer_size, hidden_size=self.hidden_layer_size, batch_first=self.batch_first)

        self.lstm_glu = GLU(
                input_size=self.hidden_layer_size,
                hidden_layer_size=self.hidden_layer_size,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                batch_first=self.batch_first)
        self.lstm_glu_add_and_norm = AddAndNorm(hidden_layer_size=self.hidden_layer_size)

        self.static_enrichment_grn = GatedResidualNetwork(
                input_size=self.hidden_layer_size,
                hidden_layer_size=self.hidden_layer_size,
                output_size=None,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                return_gate=True,
                batch_first=self.batch_first)

        # torch.Size([64, 192, 160])
        self.self_attn_layer = nn.MultiheadAttention(
                embed_dim=self.hidden_layer_size, 
                num_heads=self.num_heads, 
                dropout=self.dropout_rate, 
                # bias=False, 
                # add_bias_kv=False, 
                # add_zero_attn=False, 
                # kdim=self.hidden_layer_size // self.num_heads, 
                # vdim=self.hidden_layer_size // self.num_heads
                )
        #InterpretableMultiHeadAttention(self.num_heads, self.hidden_layer_size, dropout=self.dropout_rate)

        self.self_attention_glu = GLU(
                input_size=self.hidden_layer_size,
                hidden_layer_size=self.hidden_layer_size,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                batch_first=self.batch_first)
        self.self_attention_glu_add_and_norm = AddAndNorm(hidden_layer_size=self.hidden_layer_size)

        self.decoder_grn = GatedResidualNetwork(
                input_size=self.hidden_layer_size,
                hidden_layer_size=self.hidden_layer_size,
                output_size=None,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                return_gate=False,
                batch_first=self.batch_first)

        self.final_glu = GLU(
                input_size=self.hidden_layer_size,
                hidden_layer_size=self.hidden_layer_size,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                batch_first=self.batch_first)
        self.final_glu_add_and_norm = AddAndNorm(hidden_layer_size=self.hidden_layer_size)

        self.output_layer = LinearLayer(
                input_size=self.hidden_layer_size,
                size=self.output_size * len(self.quantiles),
                use_time_distributed=True,
                batch_first=self.batch_first)

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

        embedding_sizes = [
          self.hidden_layer_size for i, size in enumerate(self.category_counts)
        ]

        regular_inputs, categorical_inputs \
            = all_inputs[:, :, :num_regular_variables], \
              all_inputs[:, :, num_regular_variables:]

        embedded_inputs = [
            self.embeddings[i](categorical_inputs[:,:, i].long())
            for i in range(num_categorical_variables)
        ]

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
        
        # Isolate known and observed historical inputs.
        if unknown_inputs is not None:
            historical_inputs = torch.cat([
                unknown_inputs[:, :encoder_steps, :],
                known_combined_layer[:, :encoder_steps, :],
                obs_inputs[:, :encoder_steps, :]
            ], dim=-1)
        else:
            historical_inputs = torch.cat([
                  known_combined_layer[:, :encoder_steps, :],
                  obs_inputs[:, :encoder_steps, :]
              ], dim=-1)

#         print(historical_inputs)
        # Isolate only known future inputs.
        future_inputs = known_combined_layer[:, encoder_steps:, :]

        static_encoder, static_weights = self.static_combine_and_mask(static_inputs)
        static_context_variable_selection = self.static_context_variable_selection_grn(static_encoder)
        static_context_enrichment = self.static_context_enrichment_grn(static_encoder)
        static_context_state_h = self.static_context_state_h_grn(static_encoder)
        static_context_state_c = self.static_context_state_c_grn(static_encoder)
        historical_features, historical_flags, _ = self.historical_lstm_combine_and_mask(historical_inputs, static_context_variable_selection)
        future_features, future_flags, _ = self.future_lstm_combine_and_mask(future_inputs, static_context_variable_selection)

        history_lstm, (state_h, state_c) = self.lstm_encoder(historical_features, (static_context_state_h.unsqueeze(0), static_context_state_c.unsqueeze(0)))
        future_lstm, _ = self.lstm_decoder(future_features, (state_h, state_c))

        lstm_layer = torch.cat([history_lstm, future_lstm], dim=1)
        # Apply gated skip connection
        input_embeddings = torch.cat([historical_features, future_features], dim=1)

        lstm_layer, _ = self.lstm_glu(lstm_layer)
        temporal_feature_layer = self.lstm_glu_add_and_norm(lstm_layer, input_embeddings)

        # Static enrichment layers
        expanded_static_context = static_context_enrichment.unsqueeze(1)
        enriched, _ = self.static_enrichment_grn(temporal_feature_layer, expanded_static_context)

        # Decoder self attention
        mask = self.get_decoder_mask(enriched)
        x, self_att = self.self_attn_layer(enriched.permute(1,0,2), enriched.permute(1,0,2), enriched.permute(1,0,2))#, attn_mask=mask.repeat(self.num_heads, 1, 1))
        x = x.permute(1,0,2)
        # print("ATTENTION")
        # print(x.shape)
        # print(self_att.shape)
        # ATTENTION
        # (?, 192, 160)
        # (4, ?, 192, 192)

        # ATTENTION
        # torch.Size([64, 192, 160])
        # torch.Size([64, 192, 192])

        x, _ = self.self_attention_glu(x)
        x = self.self_attention_glu_add_and_norm(x, enriched)

        # Nonlinear processing on outputs
        decoder = self.decoder_grn(x)
        # Final skip connection
        decoder, _ = self.final_glu(decoder)
        transformer_layer = self.final_glu_add_and_norm(decoder, temporal_feature_layer)
        # Attention components for explainability
        attention_components = {
            # Temporal attention weights
            'decoder_self_attn': self_att,
            # Static variable selection weights
            'static_flags': static_weights[Ellipsis, 0],
            # Variable selection weights of past inputs
            'historical_flags': historical_flags[Ellipsis, 0, :],
            # Variable selection weights of future inputs
            'future_flags': future_flags[Ellipsis, 0, :]
        }

        outputs = self.output_layer(transformer_layer[:, self.num_encoder_steps:, :])
        return outputs, all_inputs, attention_components
