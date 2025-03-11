import torch
import numpy as np

from frequency_rl.utils.entropy_projection import project_entropy, project_entropy_independently

from mushroom_rl.utils.torch import TorchUtils
from mushroom_rl.policy.torch_policy import GaussianTorchPolicy

from torchaudio.functional import lfilter
from scipy.signal import butter

from torch.distributions.multivariate_normal import _batch_mv

class EntropyProjectionGaussianTorchPolicy(GaussianTorchPolicy):
    """
    Torch policy implementing a Gaussian policy with trainable standard
    deviation. The standard deviation is not state-dependent.

    """
    def __init__(self, network, input_shape, output_shape, entropy_projection_method="default", std_0=1., policy_state_shape=None, **params):
        """
        Constructor.

        Args:
            network (object): the network class used to implement the mean
                regressor;
            input_shape (tuple): the shape of the state space;
            output_shape (tuple): the shape of the action space;
            std_0 (float, 1.): initial standard deviation;
            params (dict): parameters used by the network constructor.

        """
        super().__init__(network, input_shape, output_shape, std_0=std_0, policy_state_shape=policy_state_shape, **params)
        self.entropy_projection = project_entropy_independently if entropy_projection_method == "indep" else project_entropy
        self.e_lb = self._action_dim

        self._add_save_attr(
            e_lb='primitive',
        )

    def entropy_t(self, state=None):
        _, chol_sigma = self.get_mean_and_chol(state)
        return self._action_dim / 2 * torch.log(TorchUtils.to_float_tensor(2 * np.pi * np.e))\
               + torch.sum(torch.log(torch.diag(chol_sigma)))

    def distribution_t(self, state):
        mu, chol_sigma = self.get_mean_and_chol(state)
        return torch.distributions.MultivariateNormal(loc=mu, scale_tril=chol_sigma, validate_args=False)

    def get_mean_and_chol(self, state):
        assert torch.all(torch.exp(self._log_sigma) > 0)
        chol = torch.diag(torch.exp(self._log_sigma))
        chol = self.entropy_projection(chol, self.e_lb)
        if state is None:
            return None, chol
        return self._mu(state, **self._predict_params), chol


class LowPassGaussianTorchPolicy(EntropyProjectionGaussianTorchPolicy):
    """
    Torch policy implementing a Gaussian policy with trainable standard
    deviation. The standard deviation is not state-dependent.

    """
    def __init__(self, network, input_shape, output_shape, entropy_projection_method="default",
                 cutoff_freq=None, order=1, sampling_freq=20., normalize_std=False, std_0=1., policy_state_shape=None, **params):
        """
        Constructor.

        Args:
            network (object): the network class used to implement the mean
                regressor;
            input_shape (tuple): the shape of the state space;
            output_shape (tuple): the shape of the action space;
            std_0 (float, 1.): initial standard deviation;
            params (dict): parameters used by the network constructor.

        """
        super().__init__(network, input_shape, output_shape, entropy_projection_method=entropy_projection_method,
                         std_0=std_0, policy_state_shape=policy_state_shape, **params)
        self.normalize_std = normalize_std
        lp_filter_b, lp_filter_a = butter(order, cutoff_freq, fs=sampling_freq,
                                          btype='low', analog=False)
        self.lp_filter_a = torch.tensor(lp_filter_a, dtype=torch.float32)
        self.lp_filter_b = torch.tensor(lp_filter_b, dtype=torch.float32)
        #zeros = torch.zeros((10000, self._action_dim))
        #noise_dist = torch.distributions.MultivariateNormal(loc=zeros, covariance_matrix=torch.eye(self._action_dim))
        #noise = noise_dist.sample()
        #self.noise = noise

        #noise_filtered = lfilter(noise.T, self.lp_filter_a, self.lp_filter_b, clamp=False).T
        ##scale = noise_filtered.std(1, keepdims=True)#.mean()
        #scale = noise_filtered.std(1).mean()
        #noise_filtered = noise_filtered / scale
        ##import matplotlib.pyplot as plt
        ##plt.plot(noise_filtered[:, 0].detach().numpy())
        ##plt.plot(noise_filtered[:, 1].detach().numpy())
        ##plt.show()
        ##self.noise = noise_filtered
        self.i = 0

        self._add_save_attr(
            _low_pass_noise='torch'
        )

    def reset(self):
        self.i = 0
        #zeros = torch.zeros((100, 1000, self._action_dim,))
        zeros = torch.zeros((1000, self._action_dim,))
        noise_dist = torch.distributions.MultivariateNormal(loc=zeros, covariance_matrix=torch.eye(self._action_dim))
        noise = noise_dist.sample()
        #self.noise = noise[0]

        noise_filtered = lfilter(noise.transpose(-1, -2), self.lp_filter_a, self.lp_filter_b, clamp=False).transpose(-1, -2)
        #import matplotlib.pyplot as plt
        #plt.plot(noise[0, :, 0].detach().numpy(), label="original")
        #plt.plot(noise_filtered[0, :, 0].detach().numpy(), label="filtered")
        if self.normalize_std:
            scale = noise_filtered.std(0).mean()
            noise_filtered = noise_filtered / scale
        #scale = noise_filtered.std(0).mean()
        #noise_filtered = noise_filtered / scale
        #plt.plot(noise_filtered[0, :, 0].detach().numpy(), label="filtered&scaled")
        #plt.legend()
        #plt.show()
        self.noise = noise_filtered#[0]

    def draw_action_t(self, state):
        mu, chol_sigma = self.get_mean_and_chol(state)
        #eps_ = _standard_normal(mu.shape, dtype=mu.dtype, device=mu.device)
        #eps = self.noise[self.i % self.noise.shape[0]]
        eps = self.noise[self.i]
        self.i += 1
        sample = mu + _batch_mv(chol_sigma, eps)
        return sample.detach()

    def log_prob_t(self, state, action):
        return self.distribution_t(state).log_prob(action)[:, None]

    def distribution_t(self, state):
        mu, chol_sigma = self.get_mean_and_chol(state)
        return torch.distributions.MultivariateNormal(loc=mu, scale_tril=chol_sigma, validate_args=False)

