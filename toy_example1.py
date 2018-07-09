import torch
import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# synthesize data
Ntrain = 10  #number of training spectra
Ntest = 3   #number of test spectra
L = 100  #number of freq. bins
T = 200  #number of spectra
K = 2    #number of sources

Keye = torch.eye(K)

# get train data
Xtrain = torch.zeros(Ntrain, T, L)
Mask_train = torch.zeros(Ntrain, T, L)
Mask_onehot = torch.zeros(Ntrain, L*T, K)
for n in range(Ntrain):
    mask = torch.zeros(T, L)
    for t in range(T):
        bd = int(np.mod(t, 20) + 40 + 3*np.random.randn())
        mask[t, :bd] = 1 
    Mask_train[n, :, :] = mask
    Mask_onehot[n, :, :] = Keye[mask.reshape(-1).long(), :]
    Xtrain[n, :, :] = torch.randn(T, L) * mask

# get test data
Xtest = torch.zeros(Ntest, T, L)
Mask_test = torch.zeros(Ntest, T, L)
Mask_onehot_test = torch.zeros(Ntest, L*T, K)
for n in range(Ntest):
    mask = torch.zeros(T, L)
    for t in range(T):
        bd = int(np.mod(t, 20) + 40 + 3*np.random.randn())
        mask[t, :bd] = 1 
    Mask_test[n, :, :] = mask
    Mask_onehot_test[n, :, :] = Keye[mask.reshape(-1).long(), :]
    Xtest[n, :, :] = torch.randn(T, L) * mask



# you can check the ground truth mask here
plt.matshow(mask.t()) 
plt.savefig('ground_truth_mask0.png', format='png')

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
for ep in range(EP):
    if usecuda: 
        Xtrain = Xtrain.cuda()
        Mask_onehot = Mask_onehot.cuda()
    out = mrnn.forward(Xtrain)

    #term1 = torch.matmul(out, out.permute(0, 2, 1))
    #term2 = torch.matmul(Mask_onehot, Mask_onehot.permute(0, 2, 1))

    # need to do this to save memory (outer products take a lot of space!)
    cost = 0
    for n in range(1):
        term1 = torch.matmul(out[n], out[n].t())
        term2 = torch.matmul(Mask_onehot[n], Mask_onehot[n].t())
    cost = cost + ((term1 - term2)**2).mean()

    if (ep % 100) == 0:
        plt.matshow(out[0].data.cpu().numpy().reshape(T, L, 2)[:, :, 0].transpose())
        plt.colorbar()
        plt.savefig('train_mask_output.png', format='png')

        # test 
        if usecuda:
            Xtest = Xtest.cuda()
        out_test = mrnn.forward(Xtest)

        plt.matshow(out_test[0].data.cpu().numpy().reshape(T, L, 2)[:, :, 0].transpose())
        plt.colorbar()
        plt.savefig('test_mask_output.png', format='png')

    cost.backward()
    opt.step()
    print('cost {}, epoch {}'.format(cost.item(), ep))
    #out_admats = torch.matmul(out, out.permute(0, 2, 1))


