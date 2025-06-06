"""
This file is adapted from: https://github.com/Algolzw/image-restoration-sde.
Original license: MIT (Copyright © 2023 Ziwei Luo)
Modifications: Generalized functions to take different numbers and types of inputs.
"""
import math
import torch
import abc
from tqdm import tqdm
import torchvision.utils as tvutils
import os
from scipy import integrate


class SDE(abc.ABC):
    def __init__(self, T, device=None):
        self.T = T
        self.dt = 1 / T
        self.device = device

    @abc.abstractmethod
    def drift(self, x, t):
        pass

    @abc.abstractmethod
    def dispersion(self, x, t):
        pass

    @abc.abstractmethod
    def sde_reverse_drift(self, x, score, t):
        pass

    @abc.abstractmethod
    def ode_reverse_drift(self, x, score, t):
        pass

    @abc.abstractmethod
    def score_fn(self, x, t):
        pass

    ################################################################################

    def forward_step(self, x, t):
        return x + self.drift(x, t) + self.dispersion(x, t)

    def forward_ode_step(self, x, t):
        return x + self.drift(x, t)

    def reverse_sde_step_mean(self, x, score, t):
        return x - self.sde_reverse_drift(x, score, t)

    def reverse_sde_step(self, x, score, t):
        return x - self.sde_reverse_drift(x, score, t) - self.dispersion(x, t)

    def reverse_sde_step_mu_style(self, x, score, t):
        return x - self.sde_reverse_drift_mu_style(x, score, t) - self.dispersion(x, t)

    def reverse_ode_step(self, x, score, t):
        return x - self.ode_reverse_drift(x, score, t)

    def reverse_ode_step_mu_style(self, x, score, t):
        return x - self.ode_reverse_drift_mu_style(x, score, t)

    def forward(self, x0, T=-1):
        T = self.T if T < 0 else T
        x = x0.clone()
        for t in tqdm(range(1, T + 1)):
            x = self.forward_step(x, t)

        return x

    def forward_ode(self, x0, T=-1):
        T = self.T if T < 0 else T
        x = x0.clone()
        for t in tqdm(range(1, T + 1)):
            x = self.forward_ode_step(x, t)

        return x

    def reverse_sde(self, xt, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t)
            if x.shape[1] == 4:
                x = self.reverse_sde_step(x[:, :3, :, :], score, t)
            else:
                x = self.reverse_sde_step(x, score, t)

        return x

    def reverse_ode(self, xt, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t)
            x = self.reverse_ode_step(x, score, t)

        return x


#############################################################################


class IRSDE(SDE):
    '''
    Let timestep t start from 1 to T, state t=0 is never used
    '''
    def __init__(self, max_sigma, T=100, schedule='cosine', eps=0.01,  device=None):
        super().__init__(T, device)
        self.max_sigma = max_sigma / 255 if max_sigma >= 1 else max_sigma
        self._initialize(self.max_sigma, T, schedule, eps)

    def _initialize(self, max_sigma, T, schedule, eps=0.01):

        def constant_theta_schedule(timesteps, v=1.):
            """
            constant schedule
            """
            print('constant schedule')
            timesteps = timesteps + 1 # T from 1 to 100
            return torch.ones(timesteps, dtype=torch.float32)

        def linear_theta_schedule(timesteps):
            """
            linear schedule
            """
            print('linear schedule')
            timesteps = timesteps + 1 # T from 1 to 100
            scale = 1000 / timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

        def cosine_theta_schedule(timesteps, s = 0.008):
            """
            cosine schedule
            """
            print('cosine schedule')
            timesteps = timesteps + 2 # for truncating from 1 to -1
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:-1]
            return betas

        def get_thetas_cumsum(thetas):
            return torch.cumsum(thetas, dim=0)

        def get_sigmas(thetas):
            return torch.sqrt(max_sigma**2 * 2 * thetas)

        def get_sigma_bars(thetas_cumsum):
            return torch.sqrt(max_sigma**2 * (1 - torch.exp(-2 * thetas_cumsum * self.dt)))
            
        if schedule == 'cosine':
            thetas = cosine_theta_schedule(T)
        elif schedule == 'linear':
            thetas = linear_theta_schedule(T)
        elif schedule == 'constant':
            thetas = constant_theta_schedule(T)
        else:
            print('Not implemented such schedule yet!!!')

        sigmas = get_sigmas(thetas)
        thetas_cumsum = get_thetas_cumsum(thetas) - thetas[0] # for that thetas[0] is not 0
        self.dt = -1 / thetas_cumsum[-1] * math.log(eps)
        sigma_bars = get_sigma_bars(thetas_cumsum)
        
        self.thetas = thetas.to(self.device)
        self.sigmas = sigmas.to(self.device)
        self.thetas_cumsum = thetas_cumsum.to(self.device)
        self.sigma_bars = sigma_bars.to(self.device)

        self.mu = 0.
        self.model = None

    #####################################

    # set mu for different cases
    def set_mu(self, mu):
        self.mu = mu

    def set_mu_style(self, mu_style):
        self.mu_style = mu_style

    # set score model for reverse process
    def set_model(self, model):
        self.model = model

    #####################################

    def mu_bar(self, x0, t):
        return self.mu + (x0 - self.mu) * torch.exp(-self.thetas_cumsum[t] * self.dt)

    def sigma_bar(self, t):
        return self.sigma_bars[t]

    def drift(self, x, t):
        return self.thetas[t] * (self.mu - x) * self.dt

    def sde_reverse_drift(self, x, score, t):
        return (self.thetas[t] * (self.mu - x) - self.sigmas[t]**2 * score) * self.dt

    def sde_reverse_drift_mu_style(self, x, score, t):
        return (self.thetas[t] * (self.mu_style - x) - self.sigmas[t]**2 * score) * self.dt

    def ode_reverse_drift(self, x, score, t):
        return (self.thetas[t] * (self.mu - x) - 0.5 * self.sigmas[t]**2 * score) * self.dt

    def ode_reverse_drift_mu_style(self, x, score, t):
        return (self.thetas[t] * (self.mu_style - x) - 0.5 * self.sigmas[t]**2 * score) * self.dt

    def dispersion(self, x, t):
        return self.sigmas[t] * (torch.randn_like(x) * math.sqrt(self.dt)).to(self.device)

    def get_score_from_noise(self, noise, t):
        return -noise / self.sigma_bar(t)

    def score_fn(self, x, t, **kwargs):
        # need to pre-set mu and score_model
        noise = self.model(x, self.mu, t, **kwargs)
        return self.get_score_from_noise(noise, t)

    def score_fn_mu_style(self, x, t, **kwargs):
        # need to pre-set mu and score_model
        noise = self.model(x, self.mu_style, t, **kwargs)
        return self.get_score_from_noise(noise, t)

    # These are remnants of failed experiment using zero-shot style transfer
    def score_fn_styletransfer(self, x, style_x, t, **kwargs):
        for module in self.model.modules():
            if module.__class__.__name__ == "StyleInjectionLinearAttention":
                module.store_mode = True
                module.use_style = False
        noise_style = self.model(style_x, self.mu_style, t, **kwargs)

        for module in self.model.modules():
            if module.__class__.__name__ == "StyleInjectionLinearAttention":
                module.store_mode = False
                module.use_style = True

        noise = self.model(x, self.mu, t, **kwargs)

        return self.get_score_from_noise(noise, t), self.get_score_from_noise(noise_style, t)

    def noise_fn(self, x, t, **kwargs):
        # need to pre-set mu and score_model
        return self.model(x, self.mu, t, **kwargs)

    # optimum x_{t-1}
    def reverse_optimum_step(self, xt, x0, t):
        A = torch.exp(-self.thetas[t] * self.dt)
        B = torch.exp(-self.thetas_cumsum[t] * self.dt)
        C = torch.exp(-self.thetas_cumsum[t-1] * self.dt)

        term1 = A * (1 - C**2) / (1 - B**2)
        term2 = C * (1 - A**2) / (1 - B**2)

        return term1 * (xt - self.mu) + term2 * (x0 - self.mu) + self.mu

    def reverse_optimum_std(self, t):
        A = torch.exp(-2*self.thetas[t] * self.dt)
        B = torch.exp(-2*self.thetas_cumsum[t] * self.dt)
        C = torch.exp(-2*self.thetas_cumsum[t-1] * self.dt)

        posterior_var = (1 - A) * (1 - C) / (1 - B)
        # return torch.sqrt(posterior_var)

        min_value = (1e-20 * self.dt).to(self.device)
        log_posterior_var = torch.log(torch.clamp(posterior_var, min=min_value))
        return (0.5 * log_posterior_var).exp() * self.max_sigma

    def reverse_posterior_step(self, xt, noise, t):
        x0 = self.get_init_state_from_noise(xt, noise, t)
        mean = self.reverse_optimum_step(xt, x0, t)
        std = self.reverse_optimum_std(t)
        return mean + std * torch.randn_like(xt)

    def sigma(self, t):
        return self.sigmas[t]

    def theta(self, t):
        return self.thetas[t]

    def get_real_noise(self, xt, x0, t):
        return (xt - self.mu_bar(x0, t)) / self.sigma_bar(t)

    def get_real_score(self, xt, x0, t):
        return -(xt - self.mu_bar(x0, t)) / self.sigma_bar(t)**2

    def get_init_state_from_noise(self, xt, noise, t):
        A = torch.exp(self.thetas_cumsum[t] * self.dt)
        return (xt - self.mu - self.sigma_bar(t) * noise) * A + self.mu

    # forward process to get x(T) from x(0)
    def forward(self, x0, T=-1, save_dir='forward_state'):
        T = self.T if T < 0 else T
        x = x0.clone()
        for t in tqdm(range(1, T + 1)):
            x = self.forward_step(x, t)

            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{t}.png', normalize=False)
        return x

    def reverse_sde(self, xt, T=-1, save_states=False, save_dir='sde_state', xt_style=None, condverso=False, **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        if xt_style is not None:
            sx = xt_style.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t, **kwargs)
            # The code below handles a bunch of different input configurations.
            # We are aware that it can be generalized better in future research.
            if x.shape[1] == 5:
                mask_note = xt[:, 3:4, :, :]
                mask_staff = xt[:, 4:5, :, :]
                x = self.reverse_sde_step(x[:, :3, :, :], score, t)
                x = torch.cat([x, mask_note, mask_staff], dim=1)
            elif x.shape[1] == 4:
                mask = xt[:, 3:4, :, :]
                x = self.reverse_sde_step(x[:, :3, :, :], score, t)
                x = torch.cat([x, mask], dim=1)
            elif x.shape[1] == 12:
                context = xt[:, 3:12]
                x = self.reverse_sde_step(x[:, :3, :, :], score, t)
                x = torch.cat([x, context], dim=1)
            elif x.shape[1] == 8:
                masks = xt[:, 6:8, :, :]
                x = self.reverse_sde_step(x[:, :6, :, :], score, t)
                x = torch.cat([x, masks], dim=1)
            elif condverso:
                verso_cond = xt[:, 3:6, :, :]
                x = self.reverse_sde_step(x[:, :3, :, :], score, t)
                x = torch.cat([x, verso_cond], dim=1)
            else:
                x = self.reverse_sde_step(x, score, t)
            if xt_style is not None:
                sx = self.reverse_sde_step_mu_style(sx, score_style, t)

            if save_states: # only consider to save 100 images
                interval = self.T // 100
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data[:, :3, :, :], f'{save_dir}/state_{idx}.png', normalize=False)

        return x

    def reverse_ode(self, xt, T=-1, save_states=False, save_dir='ode_state', xt_style=None, **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        if xt_style is not None:
            sx = xt_style.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score, score_style = self.score_fn_styletransfer(x, sx, t, **kwargs)
            # score = self.score_fn(x, t, **kwargs)
            x = self.reverse_ode_step(x, score, t)
            sx = self.reverse_ode_step_mu_style(sx, score_style, t)

            if save_states: # only consider to save 100 images
                interval = self.T // 100
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(sx.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x

    def reverse_posterior(self, xt, T=-1, save_states=False, save_dir='posterior_state', **kwargs):
        T = self.T if T < 0 else T

        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            noise = self.noise_fn(x, t, **kwargs)
            x = self.reverse_posterior_step(x, noise, t)

            if save_states: # only consider to save 100 images
                interval = self.T // 100
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x

    # sample ode using Black-box ODE solver (not used)
    def ode_sampler(self, xt, rtol=1e-5, atol=1e-5, method='RK45', eps=1e-3,):
        shape = xt.shape

        def to_flattened_numpy(x):
          """Flatten a torch tensor `x` and convert it to numpy."""
          return x.detach().cpu().numpy().reshape((-1,))

        def from_flattened_numpy(x, shape):
          """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
          return torch.from_numpy(x.reshape(shape))

        def ode_func(t, x):
            t = int(t)
            x = from_flattened_numpy(x, shape).to(self.device).type(torch.float32)
            score = self.score_fn(x, t)
            drift = self.ode_reverse_drift(x, score, t)
            return to_flattened_numpy(drift)

        # Black-box ODE solver for the probability flow ODE
        solution = integrate.solve_ivp(ode_func, (self.T, eps), to_flattened_numpy(xt),
                                     rtol=rtol, atol=atol, method=method)

        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(self.device).type(torch.float32)

        return x

    def optimal_reverse(self, xt, x0, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            x = self.reverse_optimum_step(x, x0, t)

        return x

    ################################################################

    def weights(self, t):
        return torch.exp(-self.thetas_cumsum[t] * self.dt)

    # sample states for training
    def generate_random_states(self, x0, mu):
        x0 = x0.to(self.device)
        mu = mu.to(self.device)

        self.set_mu(mu)

        batch = x0.shape[0]

        timesteps = torch.randint(1, self.T + 1, (batch, 1, 1, 1)).long()

        state_mean = self.mu_bar(x0, timesteps)
        noises = torch.randn_like(state_mean)
        noise_level = self.sigma_bar(timesteps)
        noisy_states = noises * noise_level + state_mean

        return timesteps, noisy_states.to(torch.float32)

    def noise_state(self, tensor):
        return tensor + torch.randn_like(tensor) * self.max_sigma




################################################################################
################################################################################
############################ Denoising SDE ##################################
################################################################################
################################################################################


class DenoisingSDE(SDE):
    '''
    Let timestep t start from 1 to T, state t=0 is never used
    '''
    def __init__(self, max_sigma, T, schedule='cosine', device=None):
        super().__init__(T, device)
        self.max_sigma = max_sigma / 255 if max_sigma > 1 else max_sigma
        self._initialize(self.max_sigma, T, schedule)

    def _initialize(self, max_sigma, T, schedule, eps=0.04):

        def linear_beta_schedule(timesteps):
            timesteps = timesteps + 1
            scale = 1000 / timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float32)

        def cosine_beta_schedule(timesteps, s = 0.008):
            """
            cosine schedule
            as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
            """
            timesteps = timesteps + 2
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype = torch.float32)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            # betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = 1 - alphas_cumprod[1:-1]
            return betas

        def get_thetas_cumsum(thetas):
            return torch.cumsum(thetas, dim=0)

        def get_sigmas(thetas):
            return torch.sqrt(max_sigma**2 * 2 * thetas)

        def get_sigma_bars(thetas_cumsum):
            return torch.sqrt(max_sigma**2 * (1 - torch.exp(-2 * thetas_cumsum * self.dt)))
        
        if schedule == 'cosine':
            thetas = cosine_beta_schedule(T)
        else:
            thetas = linear_beta_schedule(T)    
        sigmas = get_sigmas(thetas)
        thetas_cumsum = get_thetas_cumsum(thetas) - thetas[0]
        self.dt = -1 / thetas_cumsum[-1] * math.log(eps)
        sigma_bars = get_sigma_bars(thetas_cumsum)
        
        self.thetas = thetas.to(self.device)
        self.sigmas = sigmas.to(self.device)
        self.thetas_cumsum = thetas_cumsum.to(self.device)
        self.sigma_bars = sigma_bars.to(self.device)

        self.mu = 0.
        self.model = None

    # set noise model for reverse process
    def set_model(self, model):
        self.model = model

    def sigma(self, t):
        return self.sigmas[t]

    def theta(self, t):
        return self.thetas[t]

    def mu_bar(self, x0, t):
        return x0

    def sigma_bar(self, t):
        return self.sigma_bars[t]

    def drift(self, x, x0, t):
        return self.thetas[t] * (x0 - x) * self.dt

    def sde_reverse_drift(self, x, score, t):
        A = torch.exp(-2 * self.thetas_cumsum[t] * self.dt)
        return -0.5 * self.sigmas[t]**2 * (1 + A) * score * self.dt

    def ode_reverse_drift(self, x, score, t):
        A = torch.exp(-2 * self.thetas_cumsum[t] * self.dt)
        return -0.5 * self.sigmas[t]**2 * A * score * self.dt

    def dispersion(self, x, t):
        return self.sigmas[t] * (torch.randn_like(x) * math.sqrt(self.dt)).to(self.device)

    def get_score_from_noise(self, noise, t):
        return -noise / self.sigma_bar(t)

    def get_init_state_from_noise(self, x, noise, t):
        return x - self.sigma_bar(t) * noise

    def get_init_state_from_score(self, x, score, t):
        return x + self.sigma_bar(t)**2 * score

    def score_fn(self, x, t):
        # need to preset the score_model
        noise = self.model(x, t)
        return self.get_score_from_noise(noise, t)

    ############### reverse sampling ################

    def get_real_noise(self, xt, x0, t):
        return (xt - self.mu_bar(x0, t)) / self.sigma_bar(t)

    def get_real_score(self, xt, x0, t):
        return -(xt - self.mu_bar(x0, t)) / self.sigma_bar(t)**2

    def reverse_sde(self, xt, x0=None, T=-1, save_states=False, save_dir='sde_state'):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            if x0 is not None:
                score = self.get_real_score(x, x0, t)
            else:
                score = self.score_fn(x, t)
            x = self.reverse_sde_step(x, score, t)

            if save_states:
                interval = self.T // 100
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x

    def reverse_ode(self, xt, x0=None, T=-1, save_states=False, save_dir='ode_state'):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            if x0 is not None:
                real_score = self.get_real_score(x, x0, t)

            score = self.score_fn(x, t)
            x = self.reverse_ode_step(x, score, t)

            if save_states:
                interval = self.T // 100
                if t % interval == 0:
                    state = x.clone()
                    if x0 is not None:
                        state = torch.cat([x, score, real_score], dim=0)
                    os.makedirs(save_dir, exist_ok=True)
                    idx = t // interval
                    tvutils.save_image(state.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x

    def ode_sampler(self, xt, rtol=1e-5, atol=1e-5, method='RK45', eps=1e-3,):
        shape = xt.shape

        def to_flattened_numpy(x):
          """Flatten a torch tensor `x` and convert it to numpy."""
          return x.detach().cpu().numpy().reshape((-1,))

        def from_flattened_numpy(x, shape):
          """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
          return torch.from_numpy(x.reshape(shape))

        def ode_func(t, x):
            t = int(t)
            x = from_flattened_numpy(x, shape).to(self.device).type(torch.float32)
            score = self.score_fn(x, t)
            drift = self.ode_reverse_drift(x, score, t)
            return to_flattened_numpy(drift)

        # Black-box ODE solver for the probability flow ODE
        solution = integrate.solve_ivp(ode_func, (self.T, eps), to_flattened_numpy(xt),
                                     rtol=rtol, atol=atol, method=method)

        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(self.device).type(torch.float32)

        return x

    def get_optimal_timestep(self, sigma, eps=1e-6):
        sigma = sigma / 255 if sigma > 1 else sigma
        thetas_cumsum_hat = -1 / (2 * self.dt) * math.log(1 - sigma**2/self.max_sigma**2 + eps)
        T = torch.argmin((self.thetas_cumsum - thetas_cumsum_hat).abs())
        return T


    ##########################################################
    ########## below functions are used for training #########
    ##########################################################

    def reverse_optimum_step(self, xt, x0, t):
        A = torch.exp(-self.thetas[t] * self.dt)
        B = torch.exp(-self.thetas_cumsum[t] * self.dt)
        C = torch.exp(-self.thetas_cumsum[t-1] * self.dt)

        term1 = A * (1 - C**2) / (1 - B**2)
        term2 = C * (1 - A**2) / (1 - B**2)

        return term1 * (xt - x0) + term2 * (x0 - x0) + x0

    def optimal_reverse(self, xt, x0, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            x = self.reverse_optimum_step(x, x0, t)

        return x

    def weights(self, t):
        # return 0.1 + torch.exp(-self.thetas_cumsum[t] * self.dt)
        return self.sigmas[t]**2

    def generate_random_states(self, x0):
        x0 = x0.to(self.device)

        batch = x0.shape[0]
        timesteps = torch.randint(1, self.T + 1, (batch, 1, 1, 1)).long()

        noises = torch.randn_like(x0, dtype=torch.float32)
        noise_level = self.sigma_bar(timesteps)
        noisy_states = noises * noise_level + x0

        return timesteps, noisy_states

