"""
Implements Neural Processes, by Garnelo et al. 2018 by Yu Takagi
"""
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.functions import gaussian_nll
from chainer import cuda

class NeuralProcesses(chainer.Chain):
    """Neural Processes"""
    def __init__(self, n_x, n_y, n_h, n_r, n_z, activ):
        super(NeuralProcesses, self).__init__()

        # parameters
        self.n_x = n_x
        self.n_y = n_y
        self.n_h = n_h
        self.n_r = n_r
        self.n_z = n_z
        self.activ = activ

        with self.init_scope():
            # encoder
            self.le1 = L.Linear(in_size=n_x+n_y, out_size=n_h)
            self.le2 = L.Linear(in_size=n_h, out_size=n_r)

            # aggregator
            self.la_mu = L.Linear(in_size=n_r, out_size=n_z)
            self.la_ln_var = L.Linear(in_size=n_r, out_size=n_z)

            # decoder
            self.ld1 = L.Linear(in_size=n_z+n_x, out_size=n_h)
            self.ld2_mu = L.Linear(in_size=n_h, out_size=n_y)
            self.ld2_ln_var = L.Linear(in_size=n_h, out_size=n_y)

    def map_xy_to_z(self, x,y):
        # encoder
        enc_h = self.encoder(xs=x, ys=y)

        # aggregator
        z_mu, z_ln_var = self.aggregator(enc_h)

        return z_mu, z_ln_var

    def train(self, x_context, y_context, x_target, y_target):
        xp = cuda.get_array_module(x_context)
        x_all = F.concat((x_context,x_target),axis=0)
        y_all = F.concat((y_context,y_target),axis=0)

        # Map x and y to z
        z_mu_all, z_ln_var_all = self.map_xy_to_z(x=x_all,y=y_all)
        z_mu_context, z_ln_var_context = self.map_xy_to_z(x=x_context,y=y_context)
        zs = F.gaussian(z_mu_all, z_ln_var_all)
        zs_rep = F.tile(zs,(x_target.data.shape[0],1))

        # decoder
        dec_mu, dec_ln_var = self.decoder(zs=zs_rep,x_star=x_target)

        # Loss = Log-likelihood (reconstruction loss) & KL-divergence
        rec_loss = gaussian_nll(y_target, mean=dec_mu, ln_var=dec_ln_var)
        kl = _gau_kl(p_mu=z_mu_context, p_ln_var=z_ln_var_context, q_mu=z_mu_all, q_ln_var=z_ln_var_all)

        return rec_loss, kl

    def encoder(self, xs, ys):
        xp = cuda.get_array_module(xs)

        # infer latent represetation from input
        input = F.concat((xs,ys),axis=1)
        if self.activ == 'relu':
            h = F.relu(self.le1(input))
        elif self.activ == 'sigmoid':
            h = F.sigmoid(self.le1(input))

        h = self.le2(h)

        return h

    def aggregator(self, rs):
        xp = cuda.get_array_module(rs)

        # sum rs and infer mu and ln_var of latent factor
        rs = F.mean(rs,axis=0)
        rs = rs[xp.newaxis,:]
        zs_mu = self.la_mu(rs)
        zs_ln_var = self.la_ln_var(rs)

        return zs_mu, zs_ln_var

    def decoder(self, zs, x_star):
        # predict distribution
        input = F.concat((x_star,zs),axis=1)
        if self.activ == 'relu':
            h = F.relu(self.ld1(input))
        elif self.activ == 'sigmoid':
            h = F.sigmoid(self.ld1(input))
        ys_mu = self.ld2_mu(h)
        ys_ln_var = self.ld2_ln_var(h)

        return ys_mu, ys_ln_var

    def posterior(self, x_context, y_context, x_target, n_draws=1):
        xp = cuda.get_array_module(x_context)
        z_mu, z_ln_var = self.map_xy_to_z(x=x_context,y=y_context)

        dec_mus = []
        dec_ln_vars = []
        for i in range(n_draws):
            zs = F.gaussian(z_mu, z_ln_var)
            zs_rep = F.tile(zs,(x_target.data.shape[0],1))
            dec_mu, dec_ln_var = self.decoder(zs=zs_rep,x_star=x_target)
            dec_mus.append(F.transpose(dec_mu))
            dec_ln_vars.append(F.transpose(dec_ln_var))

        return dec_mus, dec_ln_vars

def _gau_kl(p_mu, p_ln_var, q_mu, q_ln_var):
    """
    Kullback-Liebler divergence from Gaussian p_mu,p_ln_var to Gaussian q_mu,q_ln_var.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    """
    p_var = F.exp(p_ln_var)
    q_var = F.exp(q_ln_var)
    # Determinants of diagonal covariances p_var, q_var
    dp_var = F.prod(p_var,axis=0)
    dq_var = F.prod(q_var,axis=0)
    # Inverse of diagonal covariance q_var
    iq_var = 1./q_var
    # Difference between means p_mu, q_mu
    diff = q_mu - p_mu
    return F.sum(0.5 *
            (F.log(dq_var / dp_var)                 # log |\Sigma_q| / |\Sigma_p|
             + F.sum(iq_var * p_var,axis=0)         # + tr(\Sigma_q^{-1} * \Sigma_p)
             + F.sum(diff * iq_var * diff,axis=0)   # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
             - len(p_mu)))                        # - N
