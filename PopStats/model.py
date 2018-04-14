import torch
import torch.nn as nn
import torch.nn.functional as F
# from loglinear import LogLinear

class DeepSet(nn.Module):

    def __init__(self, in_features, set_features=50):
        super(DeepSet, self).__init__()
        self.in_features = in_features
        self.out_features = set_features
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, 50),
            nn.ELU(inplace=True),
            nn.Linear(50, 100),
            nn.ELU(inplace=True),
            nn.Linear(100, set_features)
        )

        self.regressor = nn.Sequential(
            nn.Linear(set_features, 30),
            nn.ELU(inplace=True),
            nn.Linear(30, 30),
            nn.ELU(inplace=True),
            nn.Linear(30, 10),
            nn.ELU(inplace=True),
            nn.Linear(10, 1),
        )
        
        self.add_module('0', self.feature_extractor)
        self.add_module('1', self.regressor)
        
        
    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()
            
    def forward(self, input):
        x = input
        x = self.feature_extractor(x)
        x = x.sum(dim=1)
        x = self.regressor(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'Feature Exctractor=' + str(self.feature_extractor) \
            + '\n Set Feature' + str(self.regressor) + ')'

class DeepSet1(nn.Module):

    def __init__(self, in_features, set_features=512):
        super(DeepSet1, self).__init__()
        self.in_features = in_features
        self.out_features = set_features
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, set_features)
        )

        self.regressor = nn.Sequential(
            nn.Linear(set_features, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 1),
        )
        
        self.add_module('0', self.feature_extractor)
        self.add_module('1', self.regressor)
        
        
    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()
            
    def forward(self, input):
        x = input
        x = self.feature_extractor(x)
        x = x.sum(dim=1)
        x = self.regressor(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'Feature Exctractor=' + str(self.feature_extractor) \
            + '\n Set Feature' + str(self.regressor) + ')'


class DeepSet2(nn.Module):

    def __init__(self, in_features, set_features=256):
        super(DeepSet2, self).__init__()
        self.in_features = in_features
        self.out_features = set_features
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, set_features)
        )
        
        self.log_feature_extractor = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, set_features),
            nn.ReLU(inplace=True)
        )

        self.regressor = nn.Sequential(
            nn.Linear(set_features*2, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 1),
        )
        
        self.add_module('0', self.feature_extractor)
        self.add_module('1', self.regressor)
        
        
    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()
            
    def forward(self, input):
        x = input
        x1 = self.feature_extractor(x)
        x2 = self.log_feature_extractor(x) + 0.001
        x2 = x2.log()
        x = torch.cat((x1, x2), 2)
        x = x.sum(dim=1)
        x = self.regressor(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'Feature Exctractor=' + str(self.feature_extractor) \
            + '\n Set Feature' + str(self.regressor) + ')'

            
            
class DeepSet3(nn.Module):

    def __init__(self, in_features, set_features=50):
        super(DeepSet3, self).__init__()
        self.in_features = in_features
        self.out_features = set_features
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, 50),
            nn.ELU(inplace=True),
            nn.Linear(50, 50),
            nn.ELU(inplace=True),
            nn.Linear(50, set_features)
        )
        
        self.log_feature_extractor = nn.Sequential(
            nn.Linear(in_features, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, set_features),
            nn.ReLU(inplace=True)
        )
        
        self.l1 = nn.Linear(set_features*2, 30)
        self.l2 = LogLinear(set_features*2, 30)
        self.lp = nn.ReLU()

        self.regressor = nn.Sequential(
            #nn.Linear(set_features*2, 512),
            nn.ELU(inplace=True),
            nn.Linear(60, 30),
            nn.ELU(inplace=True),
            nn.Linear(30, 10),
            nn.ELU(inplace=True),
            nn.Linear(10, 1),
        )
        
        self.add_module('0', self.feature_extractor)
        self.add_module('1', self.regressor)
        
        
    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()
            
    def forward(self, input):
        x = input
        x1 = self.feature_extractor(x)
        x2 = self.log_feature_extractor(x) + 0.001
        x2 = x2.log()
        x = torch.cat((x1, x2), 2)
        x = x.sum(dim=1)
        x1 = self.l1(x)
        x2 = self.lp(x) + 0.001
        x2 = self.l2(x2)
        x = torch.cat((x1, x2), 1)
        x = self.regressor(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'Feature Exctractor=' + str(self.feature_extractor) \
            + '\n Set Feature' + str(self.regressor) + ')'
