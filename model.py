import math
import torch
import torch.nn as nn


class MyModelForEval(nn.Module):

    def __init__(self, window_length, num_variables, num_features):
        super(MyModelForEval, self).__init__()

        self.num_variables = num_variables
        self.num_features = num_features
        self.window_length = window_length

        self.model = MyModel(num_variables=num_variables, num_features=num_features,
                             window_length=window_length)

        self.linear = nn.Linear(num_features, window_length * num_variables)

    def forward(self, w_t):
        # e_time: [batch size, segment length, # features]
        # z: [batch size, # features]
        e_time, z = self.model(w_t, train=False)
        # z: [batch size, 1, # features]
        z = torch.unsqueeze(z, dim=1)
        # z: [batch size, 1, segment length * # variables]
        o = self.linear(z)
        # z: [batch size, segment length, # variables]
        o = o.view(z.size(0), self.window_length, self.num_variables)
        return o


class MyModelForFinetune(nn.Module):

    def __init__(self, model, window_length, num_variables, num_features):
        super(MyModelForFinetune, self).__init__()

        self.model = model

        self.window_length = window_length
        self.num_variables = num_variables
        self.num_features = num_features

        self.linear = nn.Linear(num_features, window_length * num_variables)

    def forward(self, w_t):
        # e_time: [batch size, segment length, # features]
        # z: [batch size, # features]
        e_time, z = self.model(w_t, train=False)
        # z: [batch size, 1, # features]
        z = torch.unsqueeze(z, dim=1)
        # z: [batch size, 1, segment length * # variables]
        o = self.linear(z)
        # z: [batch size, segment length, # variables]
        o = o.view(z.size(0), self.window_length, self.num_variables)
        return o


class MyModelForLinearProbing(nn.Module):

    def __init__(self, model, window_length, num_variables, num_features):
        super(MyModelForLinearProbing, self).__init__()

        self.model = model

        self.window_length = window_length
        self.num_variables = num_variables
        self.num_features = num_features

        self.linear = nn.Linear(num_features, window_length * num_variables)

        self.model.requires_grad_(False)

    def forward(self, w_t):
        # e_time: [batch size, segment length, # features]
        # z: [batch size, # features]
        e_time, z = self.model(w_t, train=False)
        # z: [batch size, 1, # features]
        z = torch.unsqueeze(z, dim=1)
        # z: [batch size, 1, segment length * # variables]
        o = self.linear(z)
        # z: [batch size, segment length, # variables]
        o = o.view(z.size(0), self.window_length, self.num_variables)
        return o


class MyModel(nn.Module):

    def __init__(self, num_variables, num_features, window_length, mask_ratio=0.0):
        super(MyModel, self).__init__()

        self.num_variables = num_variables
        self.num_features = num_features
        self.window_length = window_length
        self.mask_ratio = mask_ratio

        self.convs_time = LocalBlock(num_variables, num_features)
        self.convs_amp = LocalBlock(num_variables, num_features)
        self.convs_phase = LocalBlock(num_variables, num_features)

        self.pe_time = get_pos_encoder('fixed')(num_features, dropout=0.05, max_len=window_length)
        self.pe_amp = get_pos_encoder('fixed')(num_features, dropout=0.05, max_len=window_length)
        self.pe_phase = get_pos_encoder('fixed')(num_features, dropout=0.05, max_len=window_length)

        self.pe_fusion = get_pos_encoder('fixed')(num_features, dropout=0.05, max_len=(3 * window_length + 1))

        self.contrast_token = nn.Parameter(torch.randn(1, 1, num_features))

        self.encoder_time = Transformer(d_model=num_features, nhead=1, num_encoder_layers=2,
                                        dim_feedforward=512, dropout=0.05, activation='gelu',
                                        layer_norm_eps=1e-5, batch_first=True,
                                        device=None, dtype=None)
        self.encoder_amp = Transformer(d_model=num_features, nhead=1, num_encoder_layers=2,
                                       dim_feedforward=512, dropout=0.05, activation='gelu',
                                       layer_norm_eps=1e-5, batch_first=True,
                                       device=None, dtype=None)
        self.encoder_phase = Transformer(d_model=num_features, nhead=1, num_encoder_layers=2,
                                         dim_feedforward=512, dropout=0.05, activation='gelu',
                                         layer_norm_eps=1e-5, batch_first=True,
                                         device=None, dtype=None)

        self.encoder_fusion = Transformer(d_model=num_features, nhead=1, num_encoder_layers=2,
                                          dim_feedforward=512, dropout=0.05, activation='gelu',
                                          layer_norm_eps=1e-5, batch_first=True,
                                          device=None, dtype=None)

    def forward(self, w_time, train=True):
        # w_time: Temporal segment
        # w_time: [batch size, segment length, # variables] => [B, W, V]
        B = w_time.size(0)

        if train is False:
            w_freq = torch.fft.fft(w_time)
            w_amp, w_phase = convert_coeff(w_freq)

            w_time = self.convs_time(w_time)
            w_amp = self.convs_amp(w_amp)
            w_phase = self.convs_phase(w_phase)

            w_time = self.pe_time(w_time)
            w_amp = self.pe_amp(w_amp)
            w_phase = self.pe_phase(w_phase)

            e_time = self.encoder_time(w_time)
            e_amp = self.encoder_amp(w_amp)
            e_phase = self.encoder_phase(w_phase)

            token = self.contrast_token.repeat(B, 1, 1)

            e_cat = torch.cat((token, e_time, e_amp, e_phase), dim=1)

            e_cat = self.pe_fusion(e_cat)

            e_fused = self.encoder_fusion(e_cat)

            z = e_fused[:, 0, :]

            return e_time, z

        # w_freq: Spectral segment
        # w_amp: Spectral amplitude segment / w_phase: Spectral phase segment
        # w_amp, w_phase: [batch size, segment length, # variables] => [B, W, V]
        w_freq = torch.fft.fft(w_time)
        w_amp, w_phase = convert_coeff(w_freq)

        # w_time, w_amp, w_phase: [batch size, segment length, # features]
        w_time = self.convs_time(w_time)
        w_amp = self.convs_amp(w_amp)
        w_phase = self.convs_phase(w_phase)

        # w_time, w_amp, w_phase: [batch size, segment length, # features]
        w_time = self.pe_time(w_time)
        w_amp = self.pe_amp(w_amp)
        w_phase = self.pe_phase(w_phase)

        # w_time_{1,2}, w_amp_{1,2}, w_phase_{1,2}: [batch size, segment length, # features]
        w_time_1, w_time_2 = self._mask_for_train(w_time)
        w_amp_1, w_amp_2 = self._mask_for_train(w_amp)
        w_phase_1, w_phase_2 = self._mask_for_train(w_phase)

        # e_time_1, e_amp_1, e_phase_1}: [batch size, segment length, # features]
        e_time_1 = self.encoder_time(w_time_1)
        e_amp_1 = self.encoder_amp(w_amp_1)
        e_phase_1 = self.encoder_phase(w_phase_1)

        # e_time_2, e_amp_2, e_phase_2: [batch size, segment length, # features]
        e_time_2 = self.encoder_time(w_time_2)
        e_amp_2 = self.encoder_amp(w_amp_2)
        e_phase_2 = self.encoder_phase(w_phase_2)

        # token_{1,2}: [batch size, 1, # features]
        token_1 = self.contrast_token.repeat(B, 1, 1)
        token_2 = self.contrast_token.repeat(B, 1, 1)

        # e_cat_{1,2}: [batch size, 1 + (3 * segment length), # features]
        e_cat_1 = torch.cat((token_1, e_time_1, e_amp_1, e_phase_1), dim=1)
        e_cat_2 = torch.cat((token_2, e_time_2, e_amp_2, e_phase_2), dim=1)

        # e_cat_{1,2}: [batch size, 1 + (3 * segment length), # features]
        e_cat_1 = self.pe_fusion(e_cat_1)
        e_cat_2 = self.pe_fusion(e_cat_2)

        # e_fused_{1,2}: [batch size, 1 + (3 * segment length), # features]
        e_fused_1 = self.encoder_fusion(e_cat_1)
        e_fused_2 = self.encoder_fusion(e_cat_2)

        # z_{1,2}: [batch size, # features]
        z_1 = e_fused_1[:, 0, :]
        z_2 = e_fused_2[:, 0, :]

        return z_1, z_2

    def _mask_for_train(self, x):
        # x: [batch size, segment length, feature dims]
        def make_mask():
            m = (torch.rand((x.size(0), x.size(1)), device=x.device) < self.mask_ratio).int()
            m = 1 - m
            m = torch.unsqueeze(m, dim=-1)
            m = m.repeat(1, 1, x.size(2))
            return m
        m1 = make_mask()
        m2 = make_mask()
        x1 = x * m1
        x2 = x * m2
        return x1, x2


def convert_coeff(x, eps=1e-6):
    amp = torch.sqrt((x.real + eps).pow(2) + (x.imag + eps).pow(2))
    phase = torch.atan2(x.imag, x.real + eps)
    return amp, phase


class LocalBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(LocalBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                                kernel_size=3, dilation=2, padding=2)
        self.relu_2 = nn.ReLU()
        self.conv_3 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                                kernel_size=3, dilation=4, padding=4)

    def forward(self, x):
        # x [batch size, segment length, # features]
        x = x.transpose(1, 2)
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.conv_3(x)
        x = x.transpose(1, 2)
        return x


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        # TODO: Here, we use [batch size, sequence length, embed dim]
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        x = x.transpose(0, 1)
        return x


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        # TODO: Here, we use [batch size, sequence length, embed dim]
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        x = x.transpose(0, 1)
        return x


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))


class Transformer(nn.Module):

    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = "relu",
                 layer_norm_eps: float = 1e-5, batch_first: bool = False,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Transformer, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                   activation, layer_norm_eps, batch_first,
                                                   **factory_kwargs)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.model = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.batch_first = batch_first

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        return self.model(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform(p)
