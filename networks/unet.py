import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn


class Conv1d_Block(nn.Module):
    def __init__(self, in_filter, out_filter, inter_filter=None, kernel_size=3, use_act=False) -> None:
        super().__init__()
        
        if not inter_filter:
            inter_filter = out_filter
        padding = kernel_size//2
        
        self.conv1d = nn.Conv1d(in_channels=in_filter, out_channels=inter_filter, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(inter_filter)
        self.conv1d1 = nn.Conv1d(in_channels=inter_filter, out_channels=out_filter, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_filter)
        
        if use_act:
            self.act = nn.ReLU()
        else:
            self.act = None
    
    def forward(self, x):
        if self.act:
            x = self.act(self.conv1d(x).transpose(-1, -2))
            x = self.bn(x.transpose(-1, -2)) # back to N, C, L
            x = self.act(self.conv1d1(x).transpose(-1, -2))
            x = self.bn1(x.transpose(-1, -2)) # back to N, C, L
        else:
            x = self.bn(self.conv1d(x))
            x = self.bn1(self.conv1d1(x))
        return x
    

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels=None, scale_factor=2, block_kernel_size=3, block_act=False, mode='conv') -> None:
        super().__init__()
        if not inter_channels:
            inter_channels = out_channels
        
        if mode == 'conv':
            self.up_sampling = nn.ConvTranspose1d(in_channels=in_channels, out_channels=inter_channels, kernel_size=scale_factor, stride=scale_factor)
            self.double_conv = Conv1d_Block(in_filter=in_channels, out_filter=out_channels, inter_filter=inter_channels, kernel_size=block_kernel_size, use_act=block_act)
        else:
            self.up_sampling = nn.Upsample(scale_factor=scale_factor)
            self.double_conv = Conv1d_Block(in_filter=in_channels, out_filter=out_channels, inter_filter=inter_channels, kernel_size=block_kernel_size, use_act=block_act)
        
    def forward(self, up_layer, down_layer):
        # up_sample
        x = self.up_sampling(up_layer)
        # concat at C
        x = torch.cat((x, down_layer), dim=-2)
        # conv
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels=None, scale_factor=2, block_kernel_size=3, block_act=None) -> None:
        super().__init__()
        self.max_pool = nn.MaxPool1d(scale_factor)
        self.double_conv = Conv1d_Block(in_filter=in_channels, out_filter=out_channels, inter_filter=inter_channels, kernel_size=block_kernel_size, use_act=block_act)
    
    def forward(self, x):
        return self.double_conv(self.max_pool(x))
        

class UNet(nn.Module):
    def __init__(self, in_features=2560, conv_features=32, num_class=2, num_layers=2, scale_factor=2, kernel_size=5, use_act=True, dropout_rate=0.3, up_mode='conv') -> None:
        super(UNet, self).__init__()
        self.num_layers = num_layers
        self.scale = scale_factor
        self.kernel_size = kernel_size
        self.use_act = use_act
        
        self.dropout = nn.Dropout(dropout_rate)
        in_channels = in_features
        out_channels = conv_features
        self.layer_down = [Conv1d_Block(in_filter=in_channels, out_filter=out_channels, inter_filter=None, kernel_size=kernel_size, use_act=use_act)]
        
        # self.down
        for _ in range(num_layers):
            in_channels = out_channels
            out_channels *= scale_factor
            self.layer_down.append(Down(in_channels=in_channels, out_channels=out_channels, inter_channels=None, scale_factor=scale_factor, block_kernel_size=kernel_size, block_act=use_act))
        self.layer_down = nn.ModuleList(self.layer_down)
        
        # self.up
        self.layer_up = []
        for _ in range(num_layers):
            in_channels = out_channels
            out_channels //= scale_factor
            self.layer_up.append(Up(in_channels=in_channels, out_channels=out_channels, inter_channels=None, scale_factor=scale_factor, block_kernel_size=kernel_size, block_act=use_act, mode=up_mode))
        self.layer_up = nn.ModuleList(self.layer_up)
        
        self.out_conv = nn.Conv1d(in_channels=out_channels, out_channels=num_class, kernel_size=1)
        self.output_act = nn.Softmax(dim=-1) # assume N, L, C
        
    
    def forward(self, x):
        x = self.dropout(x.transpose(-1, -2)).transpose(-1, -2)
        self.down_x = []
        for i in range(len(self.layer_down)):
            x = self.layer_down[i](x)
            self.down_x.append(x)
        for i in range(self.num_layers):
            x = self.layer_up[i](x, self.down_x[-(i+2)])
            
        x = self.out_conv(x)
        return torch.transpose(x, -1, -2)
        

if __name__ == '__main__':
    model = UNet()
    inp = torch.randn(2, 2560, 1024)
    pred = model(inp)
    print(pred.shape)
