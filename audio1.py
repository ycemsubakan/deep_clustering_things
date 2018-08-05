import torch
import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils as ut
import argparse
import visdom
import librosa as lr

seed = 2
torch.manual_seed(seed)
np.random.seed(seed)

vis = visdom.Visdom(port=5800, server='http://cem@nmf.cs.illinois.edu', env='cem_dev')
assert vis.check_connection()

# synthesize data
Ntrain = 10  #number of training spectra
Ntest = 3   #number of test spectra
L = 513 #number of freq. bins
T = 200  #number of spectra
K = 2    #number of sources

Keye = torch.eye(K)

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch_size', type=int, default=2000, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--task', type=str, default='atomic_sourcesep',
                    help='the task name needed for the data preparation function')
parser.add_argument('--data', type=str, default='synthetic_sounds',
                    help='the dataset type')
parser.add_argument('--input_type', type=str, default='autoenc',
                    help='defines the input mode of the autoencoder')

arguments = parser.parse_args()
arguments.cuda = torch.cuda.is_available()

_, _, loader_mix = ut.preprocess_audio_files(arguments=arguments, mode='train')
_, _, loader_mix_test = ut.preprocess_audio_files(arguments=arguments, mode='test')


data = list(loader_mix)[0]
data_test = list(loader_mix_test)


class mask_rnn(nn.Module):
    def __init__(self, Krnn=100):
        super(mask_rnn, self).__init__()
        self.rnn = nn.LSTM(input_size=L, hidden_size=L, num_layers=1, 
                           bidirectional=True)
        #self.out_layer = nn.Linear(L, 2*L)

    def forward(self, x):
        h, _ = self.rnn.forward(x)
        y = F.softmax(h.view(x.size(0), -1, 2), dim=2)
        return y

mrnn = mask_rnn()

usecuda = torch.cuda.is_available()
if usecuda:
    mrnn = mrnn.cuda()

opt = optim.Adam(mrnn.parameters(), lr=1e-3)
#y_admats = torch.matmul(Mask_onehot, Mask_onehot.permute(0, 2, 1))
EP = 5000
eye = torch.eye(2)
for ep in range(EP):
    for specs, phases, _, _, _, _, _, _, mask in loader_mix:
        if usecuda: 
            specs = specs.cuda()
            mask = mask.cuda() 
            Mask_onehot = eye[mask.long(), :].reshape(mask.size(0), -1, 2).cuda()
        out = mrnn.forward(specs)

        cost = 0
        for n in range(specs.size(0)):
            norm = Mask_onehot[n].sum(0, keepdim=True)
            norm = torch.matmul(Mask_onehot[n], norm.t())
            norm = torch.sqrt(1/norm)

            term1 = -2*(torch.matmul( (out[n]*norm).t(), Mask_onehot[n])**2).mean()
            term2 = (torch.matmul( (out[n]*norm).t(), out[n])**2).mean()
     
            cost = cost + term1 + term2

        if (ep % 100) == 0:
            out_mask = (out[0].data.cpu().reshape(-1, L, 2)[:, :, 0].t() > 0.5).float()
            vis.heatmap(out_mask, win='out_map')
            
            # reconstruct the sound for training estimate for source 1
            train_hat1 = out_mask*specs[0].t().cpu()
            train_hat1_sound = lr.istft(train_hat1.numpy() * np.exp(1j*phases[0].t().numpy()))
            lr.output.write_wav('train_hat1.wav', train_hat1_sound, arguments.fs)

            # reconstruct the sound for training estimate for source 2
            train_hat2 = (1-out_mask)*specs[0].t().cpu()
            train_hat2_sound = lr.istft(train_hat2.numpy() * np.exp(1j*phases[0].t().numpy()))
            lr.output.write_wav('train_hat2.wav', train_hat2_sound, arguments.fs)

            vis.heatmap(train_hat1**.5, win='out1')
            vis.heatmap(train_hat2**.5, win='out2')

            # test 
            if usecuda:
                test_specs = data_test[0][0].cuda()
                test_phases = data_test[0][1][0].t()

            out_test = mrnn.forward(test_specs)
            out_mask_test = (out_test.data[0].reshape(-1, L, 2)[:, :, 0].cpu().t() < 0.5).float()
            
            # reconstruct the sound for test estimate for source 1
            test_hat1 = out_mask_test.numpy() * test_specs[0].cpu().t().numpy() * np.exp(1j*test_phases.numpy())
            test_hat1_sound = lr.istft(test_hat1)
            lr.output.write_wav('test_hat1.wav', test_hat1_sound, arguments.fs)

            # reconstruct the sound for test estimate for source 2
            test_hat2 = (1-out_mask_test).numpy() * test_specs[0].cpu().t().numpy() * np.exp(1j*test_phases.numpy())
            test_hat2_sound = lr.istft(test_hat2)
            lr.output.write_wav('test_hat2.wav', test_hat2_sound, arguments.fs)

            opts = {'title' : 'test out mask'}
            vis.heatmap(out_mask_test, 
                        win='out_test', opts=opts)

        cost.backward()
        opt.step()
        print('cost {}, epoch {}'.format(cost.item(), ep))
        #out_admats = torch.matmul(out, out.permute(0, 2, 1))


