import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn


class RCNN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding_set=None, with_pool=True, pool_size=2, gru_layers=2, gru_dropout=0):
        super().__init__()
        self.pool_layer = with_pool

        if padding_set is None:
            padding = kernel_size // 2
        else:
            padding = padding_set

        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        if self.pool_layer:
            self.pool = nn.MaxPool1d(pool_size)
        # self.gru = nn.GRU(input_size=out_channels, hidden_size=out_channels, num_layers=gru_layers, batch_first=True, dropout=gru_dropout)
        self.gru = nn.GRU(input_size=out_channels, hidden_size=out_channels, num_layers=gru_layers, batch_first=True, dropout=gru_dropout, bidirectional=True)

    def forward(self, x, h=None):
        sx = self.conv(x)
        if self.pool_layer:
            sx = self.pool(sx)
        tx, h = self.gru(sx.transpose(-1, -2), h)
        return torch.cat([tx.transpose(-1, -2), sx], dim=-2), h

class RCNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes=2, 
                        rcnn_kernel_size=5, rcnn_stride=1, padding_set=None, scale_factor=2, 
                        gru_layer=2, gru_dropout=0, num_rcnn_blocks=3, output_scale_factor=3, with_pool=True, dropout_rate=0.3):
        super().__init__()
        self.num_rcnn_blocks = num_rcnn_blocks
        self.pool_layer = with_pool

        blocks = []
        for i in range(self.num_rcnn_blocks):
            size = scale_factor
            if i + 1 == self.num_rcnn_blocks:
                size = output_scale_factor
            blocks.append(RCNN_Block(in_channels=in_channels, out_channels=hidden_channels, 
                                        kernel_size=rcnn_kernel_size, stride=rcnn_stride, padding_set=padding_set, pool_size=size, 
                                        gru_layers=gru_layer, gru_dropout=gru_dropout, with_pool=self.pool_layer)
                                        )
            # in_channels = hidden_channels * 2
            in_channels = hidden_channels * 3
        self.rcnn_blocks = nn.ModuleList(blocks)

        if self.pool_layer:
            up_sample = []
            # up sample to keep valid length
            if self.num_rcnn_blocks > 0:
                up_sample.append(nn.ConvTranspose1d(in_channels=in_channels, out_channels=hidden_channels, 
                                                        kernel_size=output_scale_factor, stride=output_scale_factor)
                                                        )
            if self.num_rcnn_blocks > 1:
                up2_size = scale_factor * (self.num_rcnn_blocks-1)
                up_sample.append(nn.ConvTranspose1d(in_channels=hidden_channels, out_channels=hidden_channels, 
                                                        kernel_size=up2_size, stride=up2_size)
                                                        )
            self.up = nn.ModuleList(up_sample)

            # predictor
            self.classifier = nn.Linear(in_features=hidden_channels, out_features=num_classes)
        else:
            self.classifier = nn.Linear(in_features=in_channels, out_features=num_classes)        

        self.dropout = nn.Dropout(dropout_rate)
        # self.act = nn.LeakyReLU(0.3)
        self.output_act = nn.Softmax(dim=-1)

    def forward(self, x, h=None):
        x = self.dropout(x.transpose(-1, -2)).transpose(-1, -2)
        for i in range(self.num_rcnn_blocks):
            x, h = self.rcnn_blocks[i](x, h)

        if self.pool_layer:
            # N, C, L_out=L_in
            for i in range(len(self.up)):
                x = self.up[i](x)

        # N, L_in, 2
        x = self.classifier(x.transpose(-1, -2))

        return x
        # return self.act(x)
        # return self.softmax(x)





