import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
from layers import GaussianSample

class Encoder(nn.Module):
    def __init__(self, params, sample_layer=GaussianSample):

        super(Encoder, self).__init__()

        inputDim = params.x_dim + params.y_dim
        self.bn_cat = nn.BatchNorm1d(inputDim)

        #self.fc_1 = nn.Linear(inputDim, 2000)
        #self.fc_2 = nn.Linear(2000, 500)
        #self.fc_3 = nn.Linear(500, 500)

        self.fc_1 = nn.Linear(inputDim, 256)
        self.fc_2 = nn.Linear(256, 128)
       

    def forward(self, x):

        #x = self.bn_cat(x)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        #x = F.relu(self.fc_3(x))

        return x

class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()

        inputDim = params.z_dim + params.y_dim
        
        #self.fc_1 = nn.Linear(inputDim, 500)
        #self.fc_2 = nn.Linear(500, 500)
        #self.fc_3 = nn.Linear(500, 2000)
        
        self.fc_1 = nn.Linear(inputDim, 128)
        self.fc_2 = nn.Linear(128, 256)

        #self.reconstruction = nn.Linear(2000, params.x_dim)
        self.reconstruction = nn.Linear(256, params.x_dim)

        self.output_activation = nn.Sigmoid()

    def forward(self, x):

        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        #x = F.relu(self.fc_3(x))
        x = self.reconstruction(x)

        return self.output_activation(x)

class Classifier(nn.Module):
    def __init__(self, params):

        super(Classifier, self).__init__()

        self.twoOut = params.twoOut
        self.drp_5 = nn.Dropout(.5)
        #self.fc_1 = nn.Linear(500, 1000)
        self.fc_1 = nn.Linear(params.x_dim, 256)      

        #self.logits = nn.Linear(1000, params.y_dim)
        self.logits = nn.Linear(256, params.y_dim)
        self.logitsP = nn.Linear(1000, params.y_dim)
        self.logitsN = nn.Linear(1000, params.y_dim)
        self.bn = nn.BatchNorm1d(1000)

    def forward(self, x):

        x = F.relu(self.fc_1(x))
 
        if self.twoOut:
            predsP = self.logitsP(x)
            predsN = self.logitsN(x)
            predsP = predsP.view(predsP.shape[0], predsP.shape[1], 1)
            predsN = predsN.view(predsN.shape[0], predsN.shape[1], 1)
            logits = torch.cat((predsP, predsN), dim=-1)
            preds = F.softmax(logits, dim=-1)[:,:,0]
            return logits, preds
        else:
            #x = F.softmax(self.logits(x), dim=-1)   ###################################
            x = F.sigmoid(self.logits(x))
            try:
                x1 = x.view(x.shape[0], x.shape[1], 1)
            except:
                x1 = x.view(1, x.shape[0], 1)
            # x = log(y/1 - y)
            logits = torch.cat((x1, 1 - x1), dim=-1)
            return logits, x
