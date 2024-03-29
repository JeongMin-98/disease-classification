""" Tools for implementing neural networks """
from torch import nn
from torchvision.ops import MLP


def _add_mlp_block(block_info):
    in_channel = int(block_info["in_channels"])
    hidden_channels = block_info["hidden_channels"]
    dropout_rate = float(block_info["dropout"])
    activation = block_info["activation_layer"]
    if activation == "ReLU":
        activation = nn.ReLU
    if activation == "tanh":
        activation = nn.Tanh
    block = MLP(in_channels=in_channel, hidden_channels=hidden_channels, dropout=dropout_rate,
                activation_layer=activation)
    return block


def _add_conv_block(block_info):
    in_channel = int(block_info["in_channels"])
    out_channel = int(block_info["out_channels"])
    kernel_size = int(block_info["kernel_size"])
    stride = int(block_info["stride"])
    padding = 0
    if "padding" in block_info:
        padding = int(block_info["padding"])
    return nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=0)


def _add_pooling_layer(block_info):
    kernel_size = int(block_info["kernel_size"])
    return nn.AvgPool2d(kernel_size=kernel_size, stride=2)


def set_layer(config):
    """ set layer from config file """
    module_list = nn.ModuleList()

    # input shape

    # iter config files
    for idx, info in enumerate(config):
        if info['type'] == 'MLP':
            module_list.append(_add_mlp_block(info))
            continue
        if info['type'] == 'Output':
            module_list.append(nn.LogSoftmax(dim=1))
        if info['type'] == 'Conv':
            module_list.append(_add_conv_block(info))
        if info['type'] == 'Pool':
            module_list.append(_add_pooling_layer(info))

        if "activation" in info.keys():
            if info['activation'] == "relu":
                module_list.append(nn.ReLU(inplace=True))
            if info['activation'] == "tanh":
                module_list.append(nn.Tanh(inplace=True))

    return module_list
