from torch import nn
from torchvision.ops import MLP

""" Neural Network template (Using config file) """


def _add_mlp_block(block_info):
    in_channel = int(block_info["in_channels"])
    hidden_channels = block_info["hidden_channels"]
    dropout_rate = float(block_info["dropout"])
    activation = block_info["activation_layer"]
    if activation == "ReLU":
        activation = nn.ReLU
    block = MLP(in_channels=in_channel, hidden_channels=hidden_channels, dropout=dropout_rate,
                activation_layer=activation)
    return block


def set_layer(config):
    """ set layer from config file """
    module_list = nn.ModuleList()

    # input shape

    # iter config files
    for idx, info in enumerate(config):
        if info['type'] == 'MLP':
            module_list.append(_add_mlp_block(info))
        if info['type'] == 'Output':
            module_list.append(nn.LogSoftmax(dim=1))

    return module_list


class Net(nn.Module):
    def __init__(self, config, num_classes=10):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.layers = set_layer(self.config)

    def forward(self, x):
        print(self.layers)
        x = x.view(x.size(0), -1)
        for idx, layer in enumerate(self.layers):
            x = layer(x)
        return x


class NetModel(nn.Module):
    def __init__(self, input_shape, feature_size):
        super(NetModel, self).__init__()

        self.feature_size = feature_size

        model = []
        model += [
            nn.Conv2d(in_channels=3, out_channels=self.feature_size, kernel_size=3, stride=2, padding=1, bias=True)]
        model += [nn.ReLU()]
        model += [nn.Conv2d(in_channels=self.feature_size, out_channels=self.feature_size * 2, kernel_size=3, stride=2,
                            padding=1, bias=True)]
        model += [nn.ReLU()]

        model += [nn.Flatten()]
        linear_dims = (input_shape // 4) * (input_shape // 4)
        model += [nn.Linear(in_features=linear_dims * self.feature_size * 2, out_features=10, bias=True)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x
