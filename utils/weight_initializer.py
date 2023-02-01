from torch.nn import init


class WeightInitializer:
    def __init__(self, init_type='normal', init_gain=0.02):
        self.init_type = init_type
        self.init_gain = init_gain

    def __call__(self, m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if self.init_type == 'normal':
                init.normal_(m.weight.data, 0.0, self.init_gain)
            elif self.init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=self.init_gain)
            elif self.init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif self.init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=self.init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % self.init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, self.init_gain)
            init.constant_(m.bias.data, 0.0)
