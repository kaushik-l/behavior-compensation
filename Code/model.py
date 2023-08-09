import numpy as np
import math
from math import sqrt, pi
import numpy.random as npr
import torch
from matplotlib import pyplot as plt


class Network:
    def __init__(self, name='RNN-modular', N=[40, 40], S=2, R=2, seed=1):
        self.name = name
        npr.seed(seed)
        # network parameters
        Ntot = np.sum(N)
        Nl, Nr = N
        Ml, Mr = round(Nl/4), round(Nr/4)
        self.N, self.Nl, self.Nr, self.Ml, self.Mr = Ntot, Nl, Nr, Ml, Mr  # RNN units
        self.dt = .1  # time bin (in units of tau)
        self.g_in = 1.0  # initial input weight scale
        self.g_rec = 1.05  # initial recurrent weight scale
        self.g_out = 1.0  # initial output weight scale
        self.S = S  # input
        self.R = R  # readout
        self.sig = 0  # initial activity noise
        self.z0 = np.zeros((Ntot, 1))  # initial condition
        self.xa, self.ha, self.ua = [], [], []  # input, activity, output
        self.ws = np.ones((Ntot, S))/sqrt(S)  # input weights
        self.ws[:Ml, 0] = 0
        self.ws[Ml:Nl, 1] = 0
        self.ws[Nl:(Nl+Mr), 1] = 0
        self.ws[(Nl+Mr):Ntot, 0] = 0
        self.J = self.g_rec * npr.standard_normal([Ntot, Ntot]) / np.sqrt(Ntot)  # recurrent weights
        self.J[:Nl, Nl:] = 0
        self.J[:Nl, :Nl] *= np.sqrt(Ntot) / np.sqrt(Nl)
        self.J[Nl:, :Nl] = 0
        self.J[Nl:, Nl:] *= np.sqrt(Ntot) / np.sqrt(Nr)
        self.wr = 10 * (2 * npr.random((R, Ntot)) - 1) / sqrt(Ntot)  # readout weights
        self.wr[0, Nl:] = 0
        self.wr[0, :Nl] *= np.sqrt(Ntot) / np.sqrt(Nl)
        self.wr[1, :Nl] = 0
        self.wr[1, Nl:] *= np.sqrt(Ntot) / np.sqrt(Nr)

    # nlin
    def f(self, x):
        return np.tanh(x) if not torch.is_tensor(x) else torch.tanh(x)

    # derivative of nlin
    def df(self, x):
        return 1 - (np.tanh(x) ** 2) if not torch.is_tensor(x) else 1 - (torch.tanh(x) ** 2)


class Task:
    def __init__(self, name='RDM', duration_trial=1000, dt=1,
                 n_samples=5, duration_sample=100, noise=1, signal=np.arange(-3, 4)):
        NT = int(duration_trial / dt)
        self.name, self.T, self.dt, self.NT = name, duration_trial, dt, NT
        # task parameters
        if self.name == 'RDM':
            self.noise, self.signal, self.n_samples, self.NT_sample = \
                noise, signal, n_samples, int(duration_sample / dt)
            self.s = 0.0 * np.ones((2, NT))

    def loss(self, err):
        mse = (err ** 2).mean() / 2
        return mse


class Algorithm:
    def __init__(self, name='Adam', Nepochs=10000, lr=1e-3):
        self.name = name
        # learning parameters
        self.Nepochs = Nepochs
        self.lr = lr  # learning rate
        self.Nstart_anneal = 30000
        self.annealed_lr = 1e-6  # annealed learning rate
