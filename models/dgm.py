# -*- coding: utf-8 -*-
from sklearn import linear_model as lm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from .vae import VariationalAutoencoder
from .vae import Encoder, Decoder, LadderEncoder, LadderDecoder

def weights_init(m):
    if(torch.__version__=='0.4.0'):
    	torch.nn.init.xavier_uniform_(m)
    else:
	torch.nn.init.xavier_uniform(m)
class Classifier(nn.Module):
    def __init__(self, dims, typ=0):
        """
        Single hidden layer classifier
        with softmax output.
        """
        super(Classifier, self).__init__()
        [x_dim, h_dim, y_dim] = dims
        self.type = typ
        self.drp_5 = nn.Dropout(.5)
        self.dense = nn.Linear(x_dim, int(1.5*h_dim))
        self.dense_2 = nn.Linear(int(1.5*h_dim), h_dim)
        if self.type:
            self.logits = nn.Linear(h_dim, y_dim)
        else:
            self.logitsP = nn.Linear(h_dim, y_dim)
            self.logitsN = nn.Linear(h_dim, y_dim)
        self.bn = nn.BatchNorm1d(h_dim)
    def forward(self, x):
        #x = self.drp_5(x)
        x = self.dense(x)
        x = F.relu(x)

        x = self.drp_5(x)
        x = self.dense_2(x)
        x = F.relu(x)
        #------------------------------------------------------
        if self.type:
            kk = 0 
            # x = F.sigmoid(self.logits(x))
            # x1 = x.view(x.shape[0], x.shape[1], 1)
            # logits = torch.cat((x1, 1 - x1), dim=-1)
            # return torch.log(logits+1e-8), x
        ########################################################
        else:
            predsP = self.logitsP(x)
            predsN = self.logitsN(x)
            predsP = predsP.view(predsP.shape[0], predsP.shape[1], 1)
            predsN = predsN.view(predsN.shape[0], predsN.shape[1], 1)
            logits = torch.cat((predsP, predsN), dim=-1)
            preds = F.softmax(logits, dim=-1)[:,:,0]
            return logits, preds
        ########################################################


class DeepGenerativeModel(VariationalAutoencoder):
    def __init__(self, dims, params):
        """
        M2 code replication from the paper
        'Semi-Supervised Learning with Deep Generative Models'
        (Kingma 2014) in PyTorch.

        The "Generative semi-supervised model" is a probabilistic
        model that incorporates label information in both
        inference and generation.

        Initialise a new generative model
        :param dims: dimensions of x, y, z and hidden layers.
        """
        [x_dim, self.y_dim, z_dim, h_dim] = dims
        super(DeepGenerativeModel, self).__init__([x_dim, z_dim, h_dim])

        self.encoder = Encoder([x_dim + self.y_dim, h_dim, z_dim])
        self.decoder = Decoder([z_dim + self.y_dim, list(reversed(h_dim)), x_dim])
        self.classifier = Classifier([x_dim, 600, self.y_dim], params.type)
        # self.gumbel = GumbelSoftmax(h_dim[0], self.y_dim, 10)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y):
        # Add label and data and generate latent variable
        z, z_mu, z_log_var = self.encoder(torch.cat([x, y], dim=1))

        self.kl_divergence = self._kld(z, (z_mu, z_log_var))

        # Reconstruct data point from latent data and label
        x_mu = self.decoder(torch.cat([z, y], dim=1))

        return x_mu

    def classify(self, x):
        logits = self.classifier(x)
        return logits

    def _construct_thresholds(self, probs, targets, top_k=None):

        nb_samples, nb_labels = targets.shape
        top_k = top_k or nb_labels

        # Sort predicted probabilities in descending order
        idx = np.argsort(probs, axis=1)[:,:-(top_k + 1):-1]
        p_sorted = np.vstack([probs[i, idx[i]] for i in range(len(idx))])
        t_sorted = np.vstack([targets[i, idx[i]] for i in range(len(idx))])

        # Compute F-1 measures for every possible threshold position
        F1 = []
        TP = np.zeros(nb_samples)
        FN = t_sorted.sum(axis=1)
        FP = np.zeros(nb_samples)
        for i in range(top_k):
            TP += t_sorted[:,i]
            FN -= t_sorted[:,i]
            FP += 1 - t_sorted[:,i]
            F1.append(2 * TP / (2 * TP + FN + FP))
        F1 = np.vstack(F1).T

        # Find the thresholds
        row = np.arange(nb_samples)
        col = F1.argmax(axis=1)
        p_sorted = np.hstack([p_sorted, np.zeros(nb_samples)[:, None]])
        T = 0.5 * (p_sorted[row, col] + p_sorted[row, col + 1])[:, None]

        return T

    def fit_thresholds(self, inputs, probs, Y, alpha=np.logspace(-3, 3, num=10).tolist(), cv=5, top_k=None):
      
        T = self._construct_thresholds(probs, Y)

        if isinstance(alpha, list):
            model = lm.RidgeCV(alphas=alpha, cv=cv).fit(inputs, T)
            alpha = model.alpha_
            self.t_models = lm.Ridge(alpha=alpha)
            self.t_models.fit(inputs, T)

    def predict_threshold(self, X, probs):
 
        T = self.t_models.predict(X)
        preds = probs >= T
        return preds

    def sample(self, z, y):
        """
        Samples from the Decoder to generate an x.
        :param z: latent normal variable
        :param y: label (one-hot encoded)
        :return: x
        """
        y = y.float()
        x = self.decoder(torch.cat([z, y], dim=1))
        return x


class StackedDeepGenerativeModel(DeepGenerativeModel):
    def __init__(self, dims, features):
        """
        M1+M2 model as described in [Kingma 2014].

        Initialise a new stacked generative model
        :param dims: dimensions of x, y, z and hidden layers
        :param features: a pretrained M1 model of class `VariationalAutoencoder`
            trained on the same dataset.
        """
        [x_dim, y_dim, z_dim, h_dim] = dims
        super(StackedDeepGenerativeModel, self).__init__([features.z_dim, y_dim, z_dim, h_dim])

        # Be sure to reconstruct with the same dimensions
        in_features = self.decoder.reconstruction.in_features
        self.decoder.reconstruction = nn.Linear(in_features, x_dim)

        # Make vae feature model untrainable by freezing parameters
        self.features = features
        self.features.train(False)

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        # Sample a new latent x from the M1 model
        x_sample, _, _ = self.features.encoder(x)

        # Use the sample as new input to M2
        return super(StackedDeepGenerativeModel, self).forward(x_sample, y)

    def classify(self, x):
        _, x, _ = self.features.encoder(x)
        logits = self.classifier(x)
        return logits

class AuxiliaryDeepGenerativeModel(DeepGenerativeModel):
    def __init__(self, dims):
        """
        Auxiliary Deep Generative Models [Maaløe 2016]
        code replication. The ADGM introduces an additional
        latent variable 'a', which enables the model to fit
        more complex variational distributions.

        :param dims: dimensions of x, y, z, a and hidden layers.
        """
        [x_dim, y_dim, z_dim, a_dim, h_dim] = dims
        super(AuxiliaryDeepGenerativeModel, self).__init__([x_dim, y_dim, z_dim, h_dim])

        self.aux_encoder = Encoder([x_dim, h_dim, a_dim])
        self.aux_decoder = Encoder([x_dim + z_dim + y_dim, list(reversed(h_dim)), a_dim])

        self.classifier = Classifier([x_dim + a_dim, h_dim[0], y_dim])

        self.encoder = Encoder([a_dim + y_dim + x_dim, h_dim, z_dim])
        self.decoder = Decoder([y_dim + z_dim, list(reversed(h_dim)), x_dim])

    def classify(self, x):
        # Auxiliary inference q(a|x)
        a, a_mu, a_log_var = self.aux_encoder(x)

        # Classification q(y|a,x)
        logits = self.classifier(torch.cat([x, a], dim=1))
        return logits

    def forward(self, x, y):
        """
        Forward through the model
        :param x: features
        :param y: labels
        :return: reconstruction
        """
        # Auxiliary inference q(a|x)
        q_a, q_a_mu, q_a_log_var = self.aux_encoder(x)

        # Latent inference q(z|a,y,x)
        z, z_mu, z_log_var = self.encoder(torch.cat([x, y, q_a], dim=1))

        # Generative p(x|z,y)
        x_mu = self.decoder(torch.cat([z, y], dim=1))

        # Generative p(a|z,y,x)
        p_a, p_a_mu, p_a_log_var = self.aux_decoder(torch.cat([x, y, z], dim=1))

        a_kl = self._kld(q_a, (q_a_mu, q_a_log_var), (p_a_mu, p_a_log_var))
        z_kl = self._kld(z, (z_mu, z_log_var))

        self.kl_divergence = a_kl + z_kl

        return x_mu


class LadderDeepGenerativeModel(DeepGenerativeModel):
    def __init__(self, dims):
        """
        Ladder version of the Deep Generative Model.
        Uses a hierarchical representation that is
        trained end-to-end to give very nice disentangled
        representations.

        :param dims: dimensions of x, y, z layers and h layers
            note that len(z) == len(h).
        """
        [x_dim, y_dim, z_dim, h_dim] = dims
        super(LadderDeepGenerativeModel, self).__init__([x_dim, y_dim, z_dim[0], h_dim])
	#print("="*100)
        neurons = [x_dim] +  h_dim
        encoder_layers = [LadderEncoder([neurons[i - 1], neurons[i], z_dim[i - 1]]) for i in range(1, len(neurons))]

        e = encoder_layers[-1]
        encoder_layers[-1] = LadderEncoder([e.in_features + y_dim, e.out_features, e.z_dim])

        decoder_layers = [LadderDecoder([z_dim[i - 1], h_dim[i - 1], z_dim[i]]) for i in range(1, len(h_dim))][::-1]

        self.classifier = Classifier([x_dim, h_dim[0], y_dim])

        self.encoder = nn.ModuleList(encoder_layers)
        self.decoder = nn.ModuleList(decoder_layers)
        self.reconstruction = Decoder([z_dim[0]+y_dim, h_dim, x_dim])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y):
        # Gather latent representation
        # from encoders along with final z.
        latents = []
        for i, encoder in enumerate(self.encoder):
            if i == len(self.encoder)-1:
                x, (z, mu, log_var) = encoder(torch.cat([x, y], dim=1))
            else:
                x, (z, mu, log_var) = encoder(x)
            latents.append((mu, log_var))

        latents = list(reversed(latents))

        self.kl_divergence = 0
        print("="*100)
	print(self.decoder)
	#for i, decoder in enumerate([-1 , *self.decoder]):
            # If at top, encoder == decoder,
            # use prior for KL.
        #    l_mu, l_log_var = latents[i]
        #    if i == 0:
        #        self.kl_divergence += self._kld(z, (l_mu, l_log_var))

            # Perform downword merge of information.
        #    else:
        #        z, kl = decoder(z, l_mu, l_log_var)
        #        self.kl_divergence += self._kld(*kl)

        #x_mu = self.reconstruction(torch.cat([z, y], dim=1))
        #return x_mu

    def sample(self, z, y):
        for i, decoder in enumerate(self.decoder):
            z = decoder(z)
        return self.reconstruction(torch.cat([z, y], dim=1))
