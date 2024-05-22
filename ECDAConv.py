class ZPool(nn.Module):
    def __init__(self):
        super(ZPool, self).__init__()
    def forward(self, x):
        m=torch.max(x, 1)[0].unsqueeze(1)
        n=torch.mean(x, 1).unsqueeze(1)
        return torch.cat(
            (torch.sub(m, n),n,m), dim=1
        )


class ECA_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(ECA_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


##  逆残差模块
class ECDAConv_Block(nn.Module):
    def __init__(self, inp, oup, hidden_dim, kernel_size, stride, use_eca=True):
        super(ECDAConv_Block, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        self.compress = ZPool()
        self.get_weight = nn.Sequential(nn.Conv2d(3, 1, kernel_size=3, padding=1, bias=False), nn.Sigmoid())

        # 输入通道数=扩张通道数 则不进行通道扩张
        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # ECA
                ECA_layer(oup) if use_eca else nn.Sequential(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            # 否则 先进行通道扩张
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,bias=False),
                nn.BatchNorm2d(hidden_dim),
                # ECA
                ECA_layer(oup) if use_eca else nn.Sequential(),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):

        x_out = self.compress(x)
        x_weight= self.get_weight(x_out)
        y = self.conv(x*x_weight)

        if self.identity:
            return x + y
        else:
            return y