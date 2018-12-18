# coding: utf-8
import torch
from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed
from torch import autograd


class Trainer(object):
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer,
                 gan_type, reg_type, reg_param, adaptive_beta, **kwargs):
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

        self.gan_type = gan_type
        self.reg_type = reg_type
        self.reg_param = reg_param

        self.adaptive_beta = adaptive_beta
        if self.adaptive_beta:
            self.target_kl = kwargs['target_kl']
            self.beta_step = kwargs['beta_step']

    def generator_trainstep(self, y, z):
        assert(y.size(0) == z.size(0))
        toogle_grad(self.generator, True)
        toogle_grad(self.discriminator, False)
        self.generator.train()
        self.discriminator.train()
        self.g_optimizer.zero_grad()

        x_fake = self.generator(z, y)
        d_fake = self.discriminator(x_fake, y)
        d_fake = d_fake['out']
        gloss = self.compute_loss(d_fake, 1)
        gloss.backward()

        self.g_optimizer.step()

        return gloss.item()

    def discriminator_trainstep(self, x_real, y, z):
        toogle_grad(self.generator, False)
        toogle_grad(self.discriminator, True)
        self.generator.train()
        self.discriminator.train()
        self.d_optimizer.zero_grad()

        # On real data
        x_real.requires_grad_()

        d_real_dict = self.discriminator(x_real, y)
        d_real = d_real_dict['out']
        dloss_real = self.compute_loss(d_real, 1)

        reg = 0.

        if self.reg_type == 'real' or self.reg_type == 'instnoise_real':
            dloss_real.backward(retain_graph=True)
            reg = self.reg_param * compute_grad2(d_real, x_real).mean()
            reg.backward()
        elif self.reg_type == 'vgan':
            dloss_real.backward(retain_graph=True)
            mu = d_real_dict['mu']
            logstd = d_real_dict['logstd']
            kl_real = kl_loss(mu, logstd).mean()
        elif self.reg_type == 'vgan_real':
            # Both grad penal and vgan!
            dloss_real.backward(retain_graph=True)
            # TODO: rm hard coded 10 weight for grad penal.
            reg += 10. * compute_grad2(d_real, x_real).mean()
            mu = d_real_dict['mu']
            logstd = d_real_dict['logstd']
            kl_real = kl_loss(mu, logstd).mean()
        else:
            # No reguralization.
            dloss_real.backward()

        d_acc_real = torch.mean((d_real > 0.5).float())

        # On fake data
        with torch.no_grad():
            x_fake = self.generator(z, y)

        x_fake.requires_grad_()
        d_fake_dict = self.discriminator(x_fake, y)
        d_fake = d_fake_dict['out']
        dloss_fake = self.compute_loss(d_fake, 0)

        if self.reg_type == 'fake':
            dloss_fake.backward(retain_graph=True)
            reg = self.reg_param * compute_grad2(d_fake, x_fake).mean()
            reg.backward()
        elif self.reg_type == 'vgan' or self.reg_type == 'vgan_real':
            dloss_fake.backward(retain_graph=True)
            mu_fake = d_fake_dict['mu']
            logstd_fake = d_fake_dict['logstd']
            kl_fake = kl_loss(mu_fake, logstd_fake).mean()
            avg_kl = 0.5 * (kl_real + kl_fake)
            reg += self.reg_param * avg_kl
            reg.backward()
        else:
            dloss_fake.backward()

        d_acc_fake = torch.mean((d_fake < 0.5).float())
        accuracies = {'real': d_acc_real, 'fake': d_acc_fake}

        if self.reg_type == 'wgangp':
            reg = self.reg_param * self.wgan_gp_reg(x_real, x_fake, y)
            reg.backward()

        if self.adaptive_beta:
            self.update_beta(avg_kl)

        self.d_optimizer.step()

        toogle_grad(self.discriminator, False)

        # Output
        dloss = (dloss_real + dloss_fake)

        if self.reg_type == 'none' or self.reg_type == 'instnoise':
            reg = torch.tensor(0.)

        # hack to fix div by zero
        clamp_reg_param = max(self.reg_param, 1e-5)
        reg_raw = reg / clamp_reg_param

        return dloss.item(), reg_raw.item(), accuracies

    def compute_loss(self, d_out, target):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)

        if self.gan_type == 'standard':
            loss = F.binary_cross_entropy_with_logits(d_out, targets)
        elif self.gan_type == 'wgan':
            loss = (2*target - 1) * d_out.mean()
        else:
            raise NotImplementedError

        return loss

    def wgan_gp_reg(self, x_real, x_fake, y):
        batch_size = y.size(0)
        eps = torch.rand(batch_size, device=y.device).view(batch_size, 1, 1, 1)
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out = self.discriminator(x_interp, y)
        d_out = d_out['out']

        reg = (compute_grad2(d_out, x_interp).sqrt() - 1.).pow(2).mean()

        return reg

    def update_beta(self, avg_kl):
        with torch.no_grad():
            new_beta = self.reg_param - self.beta_step * (self.target_kl - avg_kl)
            new_beta = max(new_beta, 0)
            # print('setting beta from %.2f to %.2f' % (self.reg_param, new_beta))
            self.reg_param = new_beta


# Utility functions
def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


def kl_loss(mu, logstd):
    # mu and logstd are b x k x d x d
    # make them into b*d*d x k

    dim = mu.shape[1]
    mu = mu.permute(0, 2, 3, 1).contiguous()
    logstd = logstd.permute(0, 2, 3, 1).contiguous()
    mu = mu.view(-1, dim)
    logstd = logstd.view(-1, dim)

    std = torch.exp(logstd)
    kl = torch.sum(-logstd + 0.5 * (std**2 + mu**2), dim=-1) - (0.5 * dim)

    return kl


def update_average(model_tgt, model_src, beta):
    toogle_grad(model_src, False)
    toogle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert(p_src is not p_tgt)
        p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)
