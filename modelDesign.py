# =======================================================================================================================
# =======================================================================================================================
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channel_list, H, W, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.channel_list = channel_list
        self._conv_1 = nn.Conv2d(self.channel_list[2], self.channel_list[0], kernel_size=3, padding='same')
        self._layer_norm_1 = nn.LayerNorm([self.channel_list[0], H, W])
        self._conv_2 = nn.Conv2d(self.channel_list[0], self.channel_list[1], kernel_size=3, padding='same')
        self._layer_norm_2 = nn.LayerNorm([self.channel_list[1], H, W])
        self._relu = nn.ReLU()

    def forward(self, inputs):
        x_ini = inputs
        x = self._layer_norm_1(x_ini)
        x = self._relu(x)
        x = self._conv_1(x)
        x = self._layer_norm_2(x)
        x = self._relu(x)
        x = self._conv_2(x)
        x_ini = x_ini + x
        return x_ini


class Neural_receiver(nn.Module):
    def __init__(self, subcarriers, timesymbols, streams, num_bits_per_symbol, num_blocks=6, channel_list=[24, 24, 24],
                 **kwargs):
        super(Neural_receiver, self).__init__(**kwargs)
        self.subcarriers = subcarriers
        self.timesymbols = timesymbols
        self.streams = streams
        self.num_blocks = num_blocks
        self.channel_list = channel_list
        self.num_bits_per_symbol = num_bits_per_symbol

        self.blocks = nn.Sequential()
        for block_id in range(self.num_blocks):
            block = ResBlock(channel_list=self.channel_list, H=self.timesymbols, W=self.subcarriers)
            self.blocks.add_module(name='block_{}'.format(block_id), module=block)
        self._conv_1 = nn.Conv2d(4 * self.streams, self.channel_list[2], kernel_size=3, padding='same')
        self._conv_2 = nn.Conv2d(self.channel_list[1], self.streams * self.num_bits_per_symbol, kernel_size=3,
                                 padding='same')

    def forward(self, y, template_pilot):
        # y : [batch size,NUM_LAYERS,NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS,2]
        # template_pilot : [batch size,NUM_LAYERS,NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS,2]
        batch_size = y.shape[0]
        y = y.permute(0, 2, 3, 1, 4)  # y :  [batch size,NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS,NUM_LAYERS,2]
        y = torch.reshape(y, (batch_size, self.timesymbols, self.subcarriers, self.streams * 2))
        # y :  [batch size,NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS,NUM_LAYERS*2]
        template_pilot = template_pilot.permute(0, 2, 3, 1, 4)
        # p :  [batch size,NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS,NUM_LAYERS,2]
        template_pilot = torch.reshape(template_pilot,
                                       (batch_size, self.timesymbols, self.subcarriers, self.streams * 2))

        z = torch.cat([y, template_pilot], dim=-1)
        # Channel first
        z = z.permute(0, 3, 1, 2)
        # Input conv
        z = self._conv_1(z)
        # Residual blocks
        z = self.blocks(z)
        # Output conv
        z = self._conv_2(z)
        # z :  [batch size, NUM_LAYERS*NUM_SUBCARRIERS, NUM_BITS_PER_SYMBOL, NUM_OFDM_SYMBOLS]
        # Channel last
        z = z.permute(0, 2, 3, 1)
        # z :  [batch size,NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS, NUM_LAYERS*NUM_BITS_PER_SYMBOL]
        z = torch.reshape(z, (batch_size, self.timesymbols, self.subcarriers, self.streams, self.num_bits_per_symbol))
        # z :  [batch size,NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS, NUM_LAYERS, NUM_BITS_PER_SYMBOL]
        z = z.permute(0, 3, 1, 2, 4)
        # z : [batch size, NUM_LAYERS, NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS,NUM_BITS_PER_SYMBOL)
        return z
