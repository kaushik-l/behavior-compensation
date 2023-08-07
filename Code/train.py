import numpy as np
import numpy.random as npr
import torch
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from itertools import cycle
from model import Network, Task, Algorithm
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from scipy.stats import bernoulli, norm
from scipy.ndimage.filters import uniform_filter1d


def compute_posterior(task, s):
    dt, NT, n_samples, NT_sample, noise, signal = task.dt, task.NT, task.n_samples, task.NT_sample, task.noise, task.signal
    _, idx = np.unique(s[:(n_samples * NT_sample)], return_index=True)
    m = s[:(n_samples * NT_sample)][np.sort(idx)]
    l = np.array([norm.pdf(m, loc=x, scale=noise) for x in np.concatenate((signal[signal < 0], signal[signal > 0]))])
    ln = l[:3].mean(axis=0).cumprod()               # likelihood
    ls = l[3:].mean(axis=0).cumprod()               # signal likelihood
    llr = (1/10)*np.log2(ls / ln)                   # divide by 10 to reduce the range
    ls = (1/10)*np.log2(ls)
    ln = (1/10)*np.log2(ln)
    return ls, ln, llr


def llr2belief(llr, scale):
    return 1 / ((1 / np.power(2, scale*np.array(llr))) + 1)


def ecdf(a):
    x, counts = np.unique(a, return_counts=True)
    cusum = np.cumsum(counts)
    return x, cusum / cusum[-1]


def softmax(x, lam, axis=-1):
    # save typing...
    kw = dict(axis=axis, keepdims=True)
    # make every value 0 or below, as exp(0) won't overflow
    xrel = x - x.max(**kw)
    exp_xrel = np.exp(xrel / lam)
    return exp_xrel / exp_xrel.sum(**kw)


def logit(p):
    return np.log10(p / (1 - p))


def train(net, task, algo, learningsites=('ws', 'J', 'wr'), seed=1):

    # set random seed
    npr.seed(seed)

    # convert to tensor
    sites = ('ws', 'J', 'wr')
    for site in sites:
        if site in learningsites:
            setattr(net, site, torch.tensor(getattr(net, site), requires_grad=True))
        else:
            setattr(net, site, torch.tensor(getattr(net, site), requires_grad=False))

    # optimizer
    opt = torch.optim.Adam([getattr(net, site) for site in sites], lr=algo.lr)
    lr_ = algo.lr

    # frequently used vars
    dt, NT, N, Nl, Nr, Ml, Mr, S, R = task.dt, task.NT, net.N, net.Nl, net.Nr, net.Ml, net.Mr, net.S, net.R
    n_samples, NT_sample = task.n_samples, task.NT_sample

    # track variables during learning
    learning = {'epoch': [], 'lr': [], 'mses': [], 'stim': [], 'resp': []}

    # random initialization of hidden state
    z0 = np.zeros((N, 1))    # hidden state (potential)
    net.z0 = z0  # save

    for ei in range(algo.Nepochs):

        # select context and jar
        if 'RDM' in task.name:
            stim = task.signal[npr.random_integers(len(task.signal)) - 1]
            # input
            s = task.s
            s[:, :(n_samples * NT_sample)] = stim
            n = (task.noise * npr.normal(0, 1, n_samples))
            s[:, :(n_samples * NT_sample)] += np.repeat(n, NT_sample)
            s[0, :] = [s[0, idx] if s[0, idx] >= 0 else 0 for idx in range(NT)]     # upward (+) samples
            s[1, :] = [s[1, idx] if s[1, idx] <= 0 else 0 for idx in range(NT)]     # downward (-) samples
            # target output
            ls, ln, llr = compute_posterior(task, s.sum(axis=0))
            ustar = np.zeros((NT, 2))
            ustar[:(n_samples * NT_sample), 0] = np.interp(x=np.linspace(0, n_samples, n_samples * NT_sample),
                                                           xp=range(n_samples + 1), fp=np.concatenate(([0], ls)))
            ustar[(n_samples * NT_sample):, 0] = ls[-1]
            ustar[:(n_samples * NT_sample), 1] = np.interp(x=np.linspace(0, n_samples, n_samples * NT_sample),
                                                           xp=range(n_samples + 1), fp=np.concatenate(([0], ln)))
            ustar[(n_samples * NT_sample):, 1] = ln[-1]

        # initialize activity
        z0 = net.z0     # hidden state (potential)
        h0 = net.f(z0)  # hidden state (rate)
        z, h = torch.as_tensor(z0), torch.as_tensor(h0)

        # save tensors for plotting
        sa = torch.zeros(NT, S)  # save the inputs for each time bin for plotting
        ha = torch.zeros(NT, N)  # save the hidden states for each time bin for plotting
        ua = torch.zeros(NT, R)  # save the beliefs

        # errors
        err = torch.zeros(NT, R)     # error in beliefs

        for ti in range(NT):

            # network update
            Iin = net.ws.mm(torch.as_tensor(s[:, ti][:, None]))     # input current
            Irec = net.J.mm(h)                      # recurrent current
            z = Iin + Irec                          # potential
            h = (1 - (dt/20)) * h + (dt/20) * (net.f(z))      # activity
            u = net.wr.mm(h)                        # output

            # save values for plotting
            sa[ti], ha[ti], ua[ti] = torch.as_tensor(s[:, ti]), h.T, u.T

            # error
            err[ti] = torch.tensor(ustar[ti]) - u.flatten()

        # print loss
        loss = task.loss(err)
        print('\r' + str(ei + 1) + '/' + str(algo.Nepochs) + '\t Err:' + str(loss.item()), end='')

        # save mse list and cond list
        learning['mses'].append(loss.item())
        learning['stim'].append(stim)
        learning['resp'].append(np.nanmean(ua.detach().numpy()[-20:]))

        # update learning rate if needed
        if ei >= algo.Nstart_anneal:
            lr_ *= np.exp(np.log(algo.annealed_lr / algo.lr) / (algo.Nepochs - algo.Nstart_anneal))
            opt.param_groups[0]['lr'] = lr_
            learning['lr'].append(lr_)

        # do BPTT
        loss.backward()
        opt.step()
        opt.zero_grad()

        # re-modularize RNN
        with torch.no_grad():
            net.J[:Nl, Nl:] = 0
            net.J[Nl:, :Nl] = 0
            net.wr[0, Nl:] = 0
            net.wr[1, :Nl] = 0

    return net, task, algo, learning


def test(net, task, lesion=False, lesion_module='L', Ntrials=1000, seed=1):

    # set random seed
    npr.seed(seed)

    # convert to tensor
    sites = ('ws', 'J', 'wr')
    for site in sites:
        setattr(net, site, torch.tensor(getattr(net, site), requires_grad=False))

    # frequently used vars
    dt, NT, N, Nl, Nr, Ml, Mr, S, R = task.dt, task.NT, net.N, net.Nl, net.Nr, net.Ml, net.Mr, net.S, net.R
    n_samples, NT_sample = task.n_samples, task.NT_sample

    # track variables during learning
    testing = {'stim': [], 'input': [], 'hidden': [], 'output': []}

    for ei in range(Ntrials):

        # select context and jar
        if 'RDM' in task.name:
            stim = task.signal[npr.random_integers(len(task.signal)) - 1]
            # input
            s = task.s
            s[:, :(n_samples * NT_sample)] = stim
            n = (task.noise * npr.normal(0, 1, n_samples))
            s[:, :(n_samples * NT_sample)] += np.repeat(n, NT_sample)
            s[0, :] = [s[0, idx] if s[0, idx] >= 0 else 0 for idx in range(NT)]     # upward (+) samples
            s[1, :] = [s[1, idx] if s[1, idx] <= 0 else 0 for idx in range(NT)]     # downward (-) samples

        # initialize activity
        z0 = net.z0     # hidden state (potential)
        h0 = net.f(z0)  # hidden state (rate)
        z, h = torch.as_tensor(z0), torch.as_tensor(h0)

        # save tensors for plotting
        sa = torch.zeros(NT, S)  # save the inputs
        ha = torch.zeros(NT, N)  # save the hidden states
        ua = torch.zeros(NT, R)  # save the outputs

        for ti in range(NT):

            # network update
            Iin = net.ws.mm(torch.as_tensor(s[:, ti][:, None]))     # input current
            Irec = net.J.mm(h)                      # recurrent current
            z = Iin + Irec                          # potential
            h = (1 - (dt/20)) * h + (dt/20) * (net.f(z))      # activity
            # inactivate
            if lesion and lesion_module == 'R':
                h[net.Nl:] = 0
            elif lesion and lesion_module == 'L':
                h[:net.Nl] = 0
            u = net.wr.mm(h)                        # output

            # save values for plotting
            sa[ti], ha[ti], ua[ti] = torch.as_tensor(s[:, ti]), h.T, u.T

        # save mse list and cond list
        testing['stim'].append(stim)
        testing['input'].append(sa.detach().numpy())
        testing['hidden'].append(ha.detach().numpy())
        testing['output'].append(ua.detach().numpy())

        # print trial number
        print('\r' + str(ei + 1) + '/' + str(Ntrials), end='')

    # convert to numpy array
    testing['stim'] = np.array(testing['stim'])
    testing['input'] = np.array([testing['input'][idx] for idx in range(Ntrials)])
    testing['hidden'] = np.array([testing['hidden'][idx] for idx in range(Ntrials)])
    testing['output'] = np.array([testing['output'][idx] for idx in range(Ntrials)])

    return testing


def retrain(net, task, lesion_module='R', Ntrials=1000, seed=1):

    # set random seed
    npr.seed(seed)

    # convert to tensor
    sites = ('ws', 'J', 'wr')
    for site in sites:
        setattr(net, site, torch.tensor(getattr(net, site), requires_grad=False))

    # frequently used vars
    dt, NT, N, Nl, Nr, Ml, Mr, S, R = task.dt, task.NT, net.N, net.Nl, net.Nr, net.Ml, net.Mr, net.S, net.R
    n_samples, NT_sample = task.n_samples, task.NT_sample

    # track variables during learning
    testing = {'stim': [], 'input': [], 'hidden': [], 'output': [], 'loss': []}

    for ei in range(Ntrials):

        # select context and jar
        if 'RDM' in task.name:
            stim = task.signal[npr.random_integers(len(task.signal)) - 1]
            # input
            s = task.s
            s[:, :(n_samples * NT_sample)] = stim
            n = (task.noise * npr.normal(0, 1, n_samples))
            s[:, :(n_samples * NT_sample)] += np.repeat(n, NT_sample)
            s[0, :] = [s[0, idx] if s[0, idx] >= 0 else 0 for idx in range(NT)]     # upward (+) samples
            s[1, :] = [s[1, idx] if s[1, idx] <= 0 else 0 for idx in range(NT)]     # downward (-) samples
            # target output
            ls, ln, llr = compute_posterior(task, s.sum(axis=0))
            targ = llr if lesion_module == 'R' else -llr
            ustar = np.zeros((NT, 1))
            ustar[:(n_samples * NT_sample), 0] = np.interp(x=np.linspace(0, n_samples, n_samples * NT_sample),
                                                           xp=range(n_samples + 1), fp=np.concatenate(([0], targ)))
            ustar[(n_samples * NT_sample):, 0] = targ[-1]

        # initialize activity
        z0 = net.z0     # hidden state (potential)
        h0 = net.f(z0)  # hidden state (rate)
        z, h = torch.as_tensor(z0), torch.as_tensor(h0)

        # save tensors for plotting
        sa = torch.zeros(NT, S)  # save the inputs
        ha = torch.zeros(NT, N)  # save the hidden states
        ua = torch.zeros(NT, R)  # save the outputs

        for ti in range(NT):

            # network update
            Iin = net.ws.mm(torch.as_tensor(s[:, ti][:, None]))     # input current
            Irec = net.J.mm(h)                      # recurrent current
            z = Iin + Irec                          # potential
            h = (1 - (dt/20)) * h + (dt/20) * (net.f(z))      # activity
            # inactivate
            if lesion_module == 'R':
                h[net.Nl:] = 0
            elif lesion_module == 'L':
                h[:net.Nl] = 0
            u = net.wr.mm(h)                        # output

            # save values for plotting
            sa[ti], ha[ti], ua[ti] = torch.as_tensor(s[:, ti]), h.T, u.T

        # update weights
        if lesion_module == 'R':
            err = ustar[-10:].mean() - ua[-10:, 0].mean()
            net.wr[0] += .5 * h.flatten() * err
        elif lesion_module == 'L':
            err = ustar[-10:].mean() - ua[-10:, 1].mean()
            net.wr[1] += .5 * h.flatten() * err
        testing['loss'].append(err ** 2)

        # save mse list and cond list
        testing['stim'].append(stim)
        testing['input'].append(sa.detach().numpy())
        testing['hidden'].append(ha.detach().numpy())
        testing['output'].append(ua.detach().numpy())

        # print loss
        print('\r' + str(ei + 1) + '/' + str(Ntrials) + '\t Err:' + str(testing['loss'][-1].detach().numpy()), end='')

    # convert to numpy array
    testing['stim'] = np.array(testing['stim'])
    testing['input'] = np.array([testing['input'][idx] for idx in range(Ntrials)])
    testing['hidden'] = np.array([testing['hidden'][idx] for idx in range(Ntrials)])
    testing['output'] = np.array([testing['output'][idx] for idx in range(Ntrials)])

    return testing


def plot(data):
    model = data['net']
    conds = ['prelesion', 'lesionR', 'lesionL']
    for cond in conds:
        stim, input, hidden, output = \
            data[cond]['stim'], data[cond]['input'], data[cond]['hidden'], data[cond]['output']
        output = (output[:, :, 0] - output[:, :, 1])[:, :, None]
        stims = np.unique(stim)
        Ntrials, NT = np.shape(hidden)[:2]
        N, Nl, Nr = model.N, model.Nl, model.Nr
        clrs = ['blue', 'blue', 'blue', 'Grey', 'red', 'red', 'red']
        alphas = (2/(len(stims)-1)) * np.abs(np.arange(len(stims)) - ((len(stims)-1)/2))
        alphas[int((len(stims)-1)/2)] = 1
        # plot trials
        fig = plt.figure(figsize=(16, 8), dpi=80)
        gs = fig.add_gridspec(7, 8)
        for idx in range(len(stims)):
            trlindx = (stim == stims[idx])
            fig.add_subplot(gs[idx, 0])
            plt.plot(input[trlindx, :, :].sum(axis=2).T, color=clrs[idx], alpha=alphas[idx], linewidth=.02)
            plt.plot(input[trlindx, :, :].sum(axis=2).mean(axis=0), color=clrs[idx], linewidth=2)
            plt.xticks([]), plt.yticks([]), plt.ylim((-0.5, 0.5))
            if idx == 0:
                plt.title('Input', fontsize=14)
            fig.add_subplot(gs[idx, 1])
            pca = PCA(n_components=1)
            hidden_pc = pca.fit_transform(np.reshape(hidden[:, :, :Nl], [Ntrials*NT, Nl]))
            hidden_pc = np.reshape(hidden_pc, [Ntrials, NT, 1])
            plt.plot(hidden_pc[trlindx, :, 0].T, color=clrs[idx], alpha=alphas[idx], linewidth=.02)
            plt.plot(hidden_pc[trlindx, :, 0].mean(axis=0), color=clrs[idx], alpha=alphas[idx], linewidth=2)
            plt.xticks([]), plt.yticks([]), plt.ylim(-2, 2)
            if idx == 0:
                plt.title('Left Hem.', fontsize=14)
            fig.add_subplot(gs[idx, 2])
            pca = PCA(n_components=1)
            hidden_pc = pca.fit_transform(np.reshape(hidden[:, :, Nl:], [Ntrials*NT, Nl]))
            hidden_pc = np.reshape(hidden_pc, [Ntrials, NT, 1])
            plt.plot(hidden_pc[trlindx, :, 0].T, color=clrs[idx], alpha=alphas[idx], linewidth=.02)
            plt.plot(hidden_pc[trlindx, :, 0].mean(axis=0), color=clrs[idx], alpha=alphas[idx], linewidth=2)
            plt.xticks([]), plt.yticks([]), plt.ylim(-2, 2)
            if idx == 0:
                plt.title('Right Hem.', fontsize=14)
            fig.add_subplot(gs[idx, 3])
            plt.plot(output[trlindx, :, :].sum(axis=2).T, color=clrs[idx], alpha=alphas[idx], linewidth=.02)
            plt.plot(output[trlindx, :, :].sum(axis=2).mean(axis=0), color=clrs[idx], linewidth=2)
            plt.xticks([]), plt.yticks([]), plt.ylim((-3, 3))
            if idx == 0:
                plt.title('Output', fontsize=14)
        fig.add_subplot(gs[:, 5:])
        llrs = [output[stim == stims[idx], :, :].sum(axis=2)[:, -20:].mean(axis=1).mean() for idx in range(len(stims))]
        beliefs = llr2belief(llrs, 4)
        plt.plot(stims, beliefs, 'k')
        [plt.plot(stims[idx], beliefs[idx], 'o', color=clrs[idx], alpha=alphas[idx]) for idx in range(len(stims))]
        plt.ylim((0, 1))
        plt.xlabel('Average input', fontsize=14), plt.ylabel('Posterior probability', fontsize=14)
        plt.show()
    fig = plt.figure(figsize=(8, 8), dpi=80)
    clrs = ['k', 'b', 'r']
    for cond, clr in zip(conds, clrs):
        stim, input, hidden, output = \
            data[cond]['stim'], data[cond]['input'], data[cond]['hidden'], data[cond]['output']
        output = (output[:, :, 0] - output[:, :, 1])[:, :, None]
        llrs = [output[stim == stims[idx], :, :].sum(axis=2)[:, -20:].mean(axis=1).mean() for idx in range(len(stims))]
        beliefs = llr2belief(llrs, 4)
        plt.plot(stims, beliefs, color=clr)
        plt.plot(stims, beliefs, 'o', color=clr)
    plt.ylim((0, 1))
    plt.xlabel('Average input', fontsize=14), plt.ylabel('Posterior probability', fontsize=14)
    plt.show()
