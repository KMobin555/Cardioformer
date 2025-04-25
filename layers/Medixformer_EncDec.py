import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout, activation="relu", identity=False):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(d_ff)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_ff, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(d_ff)
        self.conv3 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(d_model)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # Downsampling
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        self.identity = identity
        if not identity:
            self.residual_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
            self.residual_bn = nn.BatchNorm1d(d_model)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        
        x = self.conv1(x.transpose(-1, 1))
        x = self.bn1(x)
        x = self.activation(x)
        # x = self.pool(x)  # Apply max pooling
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.dropout(x)
        x = x.transpose(-1, 1)

        if not self.identity:
            residual = self.residual_conv(residual.transpose(-1, 1))
            residual = self.residual_bn(residual)
            residual = residual.transpose(-1, 1)
        
        x = self.norm(x + residual)
        # x = self.activation(x)
        return x

class ResNetBlock_type1(nn.Module):
    def __init__(self, d_model, d_ff, dropout, activation="relu", identity=False):
        super(ResNetBlock_type1, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        self.identity = identity
        if not self.identity:
            self.conv3 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)

    
    def forward(self, x):
        residual = x
        x = self.activation(self.conv1(x.transpose(-1, 1)))
        x = self.dropout(x)
        x = self.conv2(x).transpose(-1, 1)
        x = self.dropout(x)

        if not self.identity:
            residual = self.conv3(residual.transpose(-1, 1))
            residual = residual.transpose(-1, 1)
            residual = self.dropout(residual)

        return self.norm(residual + x)


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, dropout, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.resblock1 = ResNetBlock_type1(d_model, d_ff, dropout, activation, identity=True)
        self.resblock2 = ResNetBlock_type1(d_model, d_ff, dropout, activation, identity=True)
        self.resblock3 = ResNetBlock_type1(d_model, d_ff, dropout, activation, identity=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)  # Downsampling before attention
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = [_x + self.dropout(_nx) for _x, _nx in zip(x, new_x)]

        y = x = [self.norm1(_x) for _x in x]
        y = [self.resblock1(_y) for _y in y]
        y = [self.resblock2(_y) for _y in y]
        y = [self.resblock3(_y) for _y in y]

        # y = [self.pool(_y.transpose(-1, 1)).transpose(-1, 1) for _y in y]  # Avg pooling
        
        return [self.norm2(_x + _y) for _x, _y in zip(x, y)], attn

# class EncoderLayer(nn.Module):
#     def __init__(self, attention, d_model, d_ff, dropout, activation="relu"):
#         super(EncoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.attention = attention
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu

#     def forward(self, x, attn_mask=None, tau=None, delta=None):
#         new_x, attn = self.attention(x, attn_mask=attn_mask, tau=tau, delta=delta)
#         x = [_x + self.dropout(_nx) for _x, _nx in zip(x, new_x)]

#         y = x = [self.norm1(_x) for _x in x]
#         y = [self.dropout(self.activation(self.conv1(_y.transpose(-1, 1)))) for _y in y]
#         y = [self.dropout(self.conv2(_y).transpose(-1, 1)) for _y in y]

#         return [self.norm2(_x + _y) for _x, _y in zip(x, y)], attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [[B, L1, D], [B, L2, D], ...]

        print_pls = False
        if print_pls:
            print("(Encoder) attn layers size should be 6 and here is: ", len(self.attn_layers))
        attns = []

        if print_pls:
            print("input in (Encoder) class x: and len", x[0].shape , len(x))

        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)

            if print_pls:
                print("output in (Encoder) inside the attn_layers loop x: and len", x[0].shape , len(x))

            attns.append(attn)

        # concat all the outputs
        x = torch.cat(
            x, dim=1
        )  # (batch_size, patch_num_1 + patch_num_2 + ... , d_model)

        if print_pls:
            print("output before normalization layer inside (Encoder) x: and len", x.shape , len(x))

        if self.norm is not None:
            x = self.norm(x)

        if print_pls:
            print("output after normalization layer inside (Encoder) x: and len", x.shape , len(x))

        return x, attns
