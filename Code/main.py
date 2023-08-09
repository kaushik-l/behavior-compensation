from model import Network, Task, Algorithm
from train import train, test, retrain, plot
import numpy as np
import torch
import matplotlib

# choose simulation
train__model = False
test__model = False
retrain__model = False
plot__model = True

# simulate
if train__model:
    LIP = Network(name='RNN-modular', N=[40, 40], S=2, R=2, seed=1)
    RDM = Task(name='RDM', duration_trial=700, dt=1,
               n_samples=5, duration_sample=100, noise=0.2, signal=0.1*np.arange(-3, 4))
    ADAM = Algorithm(name='Adam', Nepochs=10000, lr=1e-3)
    sites = ('J', 'wr')
    LIP, RDM, ADAM, learning = train(net=LIP, task=RDM, algo=ADAM, learningsites=sites, seed=1)
    torch.save({'net': LIP, 'task': RDM, 'algo': ADAM, 'learning': learning},
               '..//Data//LIP__RDM.pt')

if test__model:
    data = torch.load('..//Data//LIP__RDM.pt')
    LIP, RDM, ADAM, learning = data['net'], data['task'], data['algo'], data['learning']
    prelesion = test(LIP, RDM, Ntrials=1000, seed=3)
    postlesionR = test(LIP, RDM, lesion=True, lesion_module='R', Ntrials=1000, seed=1)
    postlesionL = test(LIP, RDM, lesion=True, lesion_module='L', Ntrials=1000, seed=1)
    torch.save({'net': LIP, 'task': RDM, 'algo': ADAM, 'learning': learning, 'prelesion': prelesion,
                'lesionR': postlesionR, 'lesionL': postlesionL}, '..//Data//LIP__RDM.pt')

if retrain__model:
    data = torch.load('..//Data//LIP__RDM.pt')
    LIP, RDM, ADAM, learning, prelesion, postlesionR, postlesionL = \
        data['net'], data['task'], data['algo'], data['learning'], data['prelesion'], data['lesionR'], data['lesionL']
    LIP_lesionR, learning_lesionR = retrain(LIP, RDM, lesion_module='R', Ntrials=5000, lr=1, seed=1)
    LIP_lesionL, learning_lesionL = retrain(LIP, RDM, lesion_module='L', Ntrials=5000, lr=1, seed=1)
    torch.save({'net': LIP, 'task': RDM, 'algo': ADAM, 'learning': learning,
                'prelesion': prelesion, 'lesionR': postlesionR, 'lesionL': postlesionL,
                'net_lesionR': LIP_lesionR, 'learning_lesionR': learning_lesionR,
                'net_lesionL': LIP_lesionL, 'learning_lesionL': learning_lesionL},
               '..//Data//LIP__RDM_lesion.pt')

if plot__model:
    data_baseline = torch.load('..//Data//LIP__RDM.pt')
    data_relearn = torch.load('..//Data//LIP__RDM_lesion.pt')
    plot(data_baseline, data_relearn, 'lesion')
    plot(data_baseline, data_relearn, 'lesion_slow')
