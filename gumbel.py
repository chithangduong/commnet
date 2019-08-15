import torch 
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn 
from torch.nn import init
import pdb

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    if torch.cuda.is_available():
        U = U.cuda()
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature,beta=1.0):
    y = logits + beta*sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temp, hard=False,beta=1.0):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temp,beta)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    if torch.cuda.is_available():
        y_hard = y_hard.cuda()
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    if hard:
        return (y_hard - y).detach() + y
    else:
        return y

def gumbel_softmax2(logits, temp):
    return F.softmax(logits / temp, dim=-1)


def sinkhorn(log_alpha, n_iters=20, row=True):
    n1 = log_alpha.shape[1]
    n2 = log_alpha.shape[2]
    log_alpha = log_alpha.view(-1, n1, n2)
    for _ in range(n_iters):
        if n1 == n2:
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2).view(-1, n1, 1) # Row
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1).view(-1, 1, n2) # Col
        else:
            if row:
                log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2).view(-1, n1, 1) # Row
            else:
                log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1).view(-1, 1, n2) # Col
    return torch.exp(log_alpha)

def gumbel_sinkhorn(log_alpha, temp=1.0, n_samples=1, noise_factor=1.0, n_iters=20, squeeze=True, do_reshape=True, row=True):
    # log_alpha: BHW
    n1 = log_alpha.shape[1]
    n2 = log_alpha.shape[2]
    log_alpha = log_alpha.view(-1,n1,n2)
    batch_size = log_alpha.shape[0]
    log_alpha_w_noise = log_alpha.repeat(n_samples, 1, 1)
    if noise_factor == 0:
        noise = 0.0
    else:
        noise = sample_gumbel((n_samples*batch_size, n1, n2))*noise_factor
    # pdb.set_trace()
    log_alpha_w_noise = log_alpha_w_noise + noise
    log_alpha_w_noise = log_alpha_w_noise / temp
    sink = sinkhorn(log_alpha_w_noise, n_iters, row)
    if do_reshape:
        if n_samples > 1 or squeeze is False:
            sink = sink.view(n_samples, batch_size, n1, n2)
            sink = sink.permute(1, 0, 2, 3)
            log_alpha_w_noise = log_alpha_w_noise.view(n_samples, batch_size, n1, n2)
            log_alpha_w_noise = log_alpha_w_noise.permute(1, 0, 2, 3)
    return sink