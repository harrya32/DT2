import unittest
import hydra
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from torch import multiprocessing
import os

import numpy as np
import random
from collections import defaultdict
import time

import os
import random
import time
import traceback
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from functools import partial
from copy import deepcopy
from enum import Enum
from scipy.stats import truncnorm 


import atexit
import click
import datetime
import os
import requests
import sys
import yaml
import json
import openai
from functools import partial
from collections import deque
from scipy.optimize import minimize
import math
import jax
import jax.numpy as jnp
from jax import random, vmap, jit, lax

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from pathlib import Path

OFFLINE_CONTROL_NPZ_FILES = {
    'Dataset-Pendulum': 'pendulum_data.npz',
    'Dataset-LunarLander': 'lunarlander_data.npz',
    'Dataset-Walker': 'walker_data.npz',
    'Dataset-Hopper': 'hopper_data.npz',
    'Dataset-Cheetah': 'cheetah_data.npz',
    'Dataset-Ant': 'ant_data.npz',
}

# Pendulum keeps scalar-per-dimension arguments for backward compatibility.
OFFLINE_CONTROL_GENERIC_VECTOR_ENVS = {
    env_name for env_name in OFFLINE_CONTROL_NPZ_FILES if env_name != 'Dataset-Pendulum'
}

def dict_to_array(constants_dict):
    """
    Convert a dictionary of constants to an array.

    Parameters:
    constants_dict (dict): Dictionary of constants.

    Returns:
    list: List of constant values.
    """
    # Ensure consistent ordering of keys
    keys = sorted(constants_dict.keys())
    return [constants_dict[key] for key in keys]

def array_to_dict(constants_array, template_dict):
    """
    Convert an array of constants back to a dictionary.

    Parameters:
    constants_array (list): List of constant values.
    template_dict (dict): A template dictionary to get the keys.

    Returns:
    dict: Dictionary of constants.
    """
    keys = sorted(template_dict.keys())
    return {key: value for key, value in zip(keys, constants_array)}

def generate_bounds(param_dict):
    bounds = []
    keys = sorted(param_dict.keys())
    for key in keys:
        value = param_dict[key]
        # Determine the order of magnitude
        order_of_magnitude = 10 ** (int(math.log10(abs(value))) + 1)

        # # Lower bound is an order of magnitude less, unless the value is 0
        # lower_bound = max(value - order_of_magnitude, 0) if value != 0 else 0

        # # Upper bound is an order of magnitude more
        # upper_bound = value + order_of_magnitude

        # Append the tuple of bounds to the list
        # bounds.append((lower_bound, upper_bound))
        bounds.append((-order_of_magnitude, order_of_magnitude))

    return bounds

# True env
probabilistic = False
device = "cuda:0"

def get_model_parameters(model):
    param_dict = {}
    for name, param in model.named_parameters():
        if '.' not in name:
            param_cpu = param.detach().cpu()
            if param_cpu.numel() == 1:
                param_dict[name] = param_cpu.item()
            else:
                param_dict[name] = param_cpu.tolist()
    return param_dict

def build_transition_windows(states, actions, next_states, done, window_length):
    terminal_indices = np.where(done > 0.5)[0]
    if terminal_indices.size == 0 or terminal_indices[-1] != (states.shape[0] - 1):
        terminal_indices = np.concatenate([terminal_indices, np.array([states.shape[0] - 1])])
    episode_starts = np.concatenate([np.array([0]), terminal_indices[:-1] + 1])

    state_windows = []
    action_windows = []
    for start_idx, end_idx in zip(episode_starts, terminal_indices):
        episode_states = states[start_idx:end_idx + 1]
        episode_actions = actions[start_idx:end_idx + 1]
        if episode_states.shape[0] == 0:
            continue
        final_next_state = next_states[end_idx:end_idx + 1]
        episode_states = np.concatenate([episode_states, final_next_state], axis=0)
        episode_actions = np.concatenate([episode_actions, episode_actions[-1:]], axis=0)
        if episode_states.shape[0] < window_length:
            continue
        for t in range(episode_states.shape[0] - window_length + 1):
            state_windows.append(episode_states[t:t + window_length])
            action_windows.append(episode_actions[t:t + window_length])

    if len(state_windows) == 0:
        raise ValueError('No trajectory windows were created. Check data or reduce run.dataset_window_length.')

    state_windows = np.stack(state_windows, axis=0)
    action_windows = np.stack(action_windows, axis=0)
    return state_windows, action_windows

class DatasetEnv:
    def __init__(self):
        pass

    def reset(self, num_patients=1):
        pass
    
    def evaluate_simulator_code_wrapper(self, StateDifferential, train_data, val_data, test_data, config={}, logger=None, env_name='', env_seed=0):
        if config.run.optimizer == 'pytorch':
            train_loss, val_loss, optimized_parameters, loss_per_dim, test_loss, checkpoint_path = self.evaluate_simulator_code_using_pytorch(StateDifferential, train_data, val_data, test_data, config=config, logger=logger, env_name=env_name, env_seed=env_seed)
        elif 'evotorch' in config.run.optimizer:
            train_loss, val_loss, optimized_parameters, loss_per_dim, test_loss = self.evaluate_simulator_code_using_pytorch_with_neuroevolution(StateDifferential, train_data, val_data, test_data, config=config, logger=logger)
            checkpoint_path = ''
        if env_name == 'Dataset-3DLV':
            loss_per_dim_dict = {'prey_population': loss_per_dim[0], 'intermediate_population': loss_per_dim[1], 'top_predators_population': loss_per_dim[2]}
        elif env_name == 'Dataset-HL':
            loss_per_dim_dict = {'hare_population': loss_per_dim[0], 'lynx_population': loss_per_dim[1]}
        elif env_name == 'Dataset-Pendulum':
            loss_per_dim_dict = {'cos_theta': loss_per_dim[0], 'sin_theta': loss_per_dim[1], 'theta_dot': loss_per_dim[2]}
        else:
            loss_per_dim_dict = {f'state_{idx}': value for idx, value in enumerate(loss_per_dim)}
        return train_loss, val_loss, optimized_parameters, loss_per_dim_dict, test_loss, checkpoint_path
    
    def evaluate_simulator_code_using_pytorch(self, StateDifferential, train_data, val_data, test_data, config={}, logger=None, env_name='', env_seed=0):
        import torch
        import numpy as np
        use_cuda = bool(config.setup.cuda) and torch.cuda.is_available()
        if bool(config.setup.cuda) and not torch.cuda.is_available() and logger is not None:
            logger.info("[WARNING] CUDA requested but not available. Falling back to CPU.")
        device = "cuda:0" if use_cuda else "cpu"

        # Wrap in try
        f_model = StateDifferential()
        f_model.to(device)

        f_model.train()
        states_train, actions_train = train_data
        if actions_train is not None:
            actions_train = torch.tensor(actions_train, dtype=torch.float32, device=device)
        states_train = torch.tensor(states_train, dtype=torch.float32, device=device)
        requested_batch_size = int(config.run.pytorch_as_optimizer.batch_size)
        if requested_batch_size < 1:
            if logger is not None:
                logger.info(f"[WARNING] Invalid batch_size={requested_batch_size}; using 1.")
            requested_batch_size = 1
        batch_size = min(requested_batch_size, states_train.shape[0])
        config.run.pytorch_as_optimizer.batch_size = batch_size

        states_val, actions_val = val_data
        if actions_val is not None:
            actions_val = torch.tensor(actions_val, dtype=torch.float32, device=device)
        states_val = torch.tensor(states_val, dtype=torch.float32, device=device)

        MSE = torch.nn.MSELoss()
        optimizer = optim.Adam(f_model.parameters(), lr=config.run.pytorch_as_optimizer.learning_rate, weight_decay=config.run.pytorch_as_optimizer.weight_decay)
        # clip_grad_norm = config.run.clip_grad_norm if config.run.clip_grad_norm > 0 else None

        def coerce_model_output_to_dx_dt(model_output, expected_state_dim):
            if isinstance(model_output, torch.Tensor):
                if model_output.ndim == 1:
                    model_output = model_output.unsqueeze(-1)
                if model_output.ndim != 2:
                    raise ValueError(f'Expected model output tensor with 2 dims, got shape {tuple(model_output.shape)}')
                dx_dt = model_output
            elif isinstance(model_output, (tuple, list)):
                if len(model_output) == 0:
                    raise ValueError('Model output tuple/list is empty.')
                output_columns = []
                for col in model_output:
                    if not isinstance(col, torch.Tensor):
                        raise TypeError(f'Model output contains non-tensor element of type {type(col)}')
                    if col.ndim == 1:
                        col = col.unsqueeze(-1)
                    if col.ndim != 2 or col.shape[1] != 1:
                        raise ValueError(f'Each model output element must have shape (batch,) or (batch,1), got {tuple(col.shape)}')
                    output_columns.append(col)
                dx_dt = torch.cat(output_columns, dim=-1)
            else:
                raise TypeError(f'Unsupported model output type: {type(model_output)}')

            if dx_dt.shape[1] != expected_state_dim:
                raise ValueError(f'Output state dimension mismatch. Expected {expected_state_dim}, got {dx_dt.shape[1]}')
            return dx_dt

        def compute_dx_dt(model, states_at_t, actions_at_t):
            if env_name == 'Dataset-3DLV':
                prey_population, intermediate_population, top_predators_population = states_at_t[:,0], states_at_t[:,1], states_at_t[:,2]
                model_output = model(prey_population, intermediate_population, top_predators_population)
            elif env_name == 'Dataset-HL':
                hare, lynx, time = states_at_t[:,0], states_at_t[:,1], actions_at_t[:,0]
                model_output = model(hare, lynx, time)
            elif env_name == 'Dataset-Pendulum':
                cos_theta, sin_theta, theta_dot = states_at_t[:,0], states_at_t[:,1], states_at_t[:,2]
                torque = actions_at_t[:,0]
                model_output = model(cos_theta, sin_theta, theta_dot, torque)
            elif env_name in OFFLINE_CONTROL_GENERIC_VECTOR_ENVS:
                if actions_at_t is None:
                    raise ValueError(f'Actions are required for {env_name}.')
                model_output = model(states_at_t, actions_at_t)
            else:
                raise NotImplementedError(f'Unsupported dataset env for pytorch simulator evaluation: {env_name}')
            return coerce_model_output_to_dx_dt(model_output, expected_state_dim=states_at_t.shape[1])

        def train(model, states_train_batch_i, actions_train_batch_i):
            optimizer.zero_grad(True)
            pred_states = []
            pred_state = states_train_batch_i[:,0]
            for t in range(states_train_batch_i.shape[1]):
                pred_states.append(pred_state)
                states_at_t = states_train_batch_i[:,t]
                actions_at_t = actions_train_batch_i[:,t] if actions_train_batch_i is not None else None
                dx_dt = compute_dx_dt(model, states_at_t, actions_at_t)
                pred_state = states_train_batch_i[:,t] + dx_dt
                # pred_state[pred_state<=0] = 0
            pred_states = torch.stack(pred_states, dim=1)
            loss = MSE(pred_states, states_train_batch_i)
            loss.backward()
            # if clip_grad_norm:
            #     torch.nn.utils.clip_grad_norm_(f_model.parameters(), clip_grad_norm)
            optimizer.step()
            return loss.detach()
        
        # train_opt = torch.compile(train)
        # train_opt = torch.compile(train)
        train_opt = train

        def compute_eval_loss(model, dataset):
            states, actions = dataset
            model.eval()
            with torch.no_grad():
                pred_states = []
                # pred_sates_per_dim_per_bb = []
                pred_state = states[:,0]
                for t in range(states.shape[1]):
                    pred_states.append(pred_state)
                    states_at_t = states[:,t]
                    actions_at_t = actions[:,t] if actions is not None else None
                    dx_dt = compute_dx_dt(model, states_at_t, actions_at_t)
                    pred_state = states[:,t] + dx_dt
                pred_states = torch.stack(pred_states, dim=1)
                val_loss = MSE(pred_states, states).item()
                loss_per_dim = torch.mean(torch.square(pred_states - states), dim=(0,1)).cpu().tolist()
            model.train()
            return val_loss, loss_per_dim
                
        best_model = None
        if config.run.optimize_params:
            best_val_loss = float('inf')  # Initialize with a very high value
            patience_counter = 0  # Counter for tracking patience

            for epoch in range(config.run.pytorch_as_optimizer.epochs):
                iters = 0 
                cum_loss = torch.zeros((), device=device)
                t0 = time.perf_counter()
                permutation = torch.randperm(states_train.shape[0])
                for batch_start in range(0, permutation.shape[0], batch_size):
                    indices = permutation[batch_start:batch_start + batch_size]
                    states_train_batch = states_train[indices]
                    if actions_train is not None:
                        actions_train_batch = actions_train[indices]
                    else:
                        actions_train_batch = None
                    cum_loss += train_opt(f_model, states_train_batch, actions_train_batch)
                    iters += 1
                time_taken = time.perf_counter() - t0
                if epoch % config.run.pytorch_as_optimizer.log_interval == 0:
                    # Collect validation loss
                    val_loss, _ = compute_eval_loss(f_model, (states_val, actions_val))
                    train_loss_epoch = (cum_loss / max(iters, 1)).item()
                    print(f'[EPOCH {epoch} COMPLETE] MSE TRAIN LOSS {train_loss_epoch:.4f} | MSE VAL LOSS {val_loss:.4f} | s/epoch: {time_taken:.2f}s')
                    # Early stopping check
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model = deepcopy(f_model.state_dict())
                        patience_counter = 0  # Reset counter on improvement
                    else:
                        patience_counter += 1  # Increment counter if no improvement
                    if patience_counter >= config.run.optimization.patience:
                        logger.info(f"Early stopping triggered at epoch {epoch}")
                        break  # Exit the loop if no improvement for 'patience' generations
        else:
            cum_loss, iters = torch.tensor(1.0, device=device), 1

        # Save model after training
        f_model.eval()
        if best_model is not None:
            f_model.load_state_dict(best_model)
            print('Loaded best model')
            
        val_loss, _ = compute_eval_loss(f_model, (states_val, actions_val))
        # torch.save(f_model.state_dict(), f'{folder_path}dynode_model_{env.env_name}_{env.seed}.pt')
        print(f'[Train Run completed successfully] MSE VAL LOSS {val_loss:.4f}')
        print('')

        val_loss, loss_per_dim = compute_eval_loss(f_model, (states_val, actions_val))
        train_loss = float((cum_loss / max(iters, 1)).item())
        optimized_parameters = get_model_parameters(f_model)

        states_test, actions_test = test_data
        if actions_test is not None:
            actions_test = torch.tensor(actions_test, dtype=torch.float32, device=device)
        states_test = torch.tensor(states_test, dtype=torch.float32, device=device)
        test_data = (states_test, actions_test)

        test_loss, _ = compute_eval_loss(f_model, test_data)

        # Save full model weights so NN parameters can be restored later.
        base_log_folder = config.run.log_path.split('.txt')[0]
        checkpoint_folder = Path(f'{base_log_folder}/{env_name}/{env_seed}/candidate_state_dicts')
        checkpoint_folder.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_folder / f'candidate_{time.time_ns()}.pt'
        torch.save(f_model.state_dict(), checkpoint_path)

        return train_loss, val_loss, optimized_parameters, loss_per_dim, test_loss, str(checkpoint_path)
    

def load_data(config={}, seed=0, env_name='', train_ratio=0.7, val_ratio=0.15):
    def cfg_get(path, default):
        try:
            current = config
            for key in path.split('.'):
                if isinstance(current, dict):
                    current = current[key]
                else:
                    current = getattr(current, key)
            return current
        except Exception:
            return default

    if env_name == 'Dataset-3DLV':
        pandas_csv_path = './libs/datasets/data/TS_3DLV.csv'
        df = pd.read_csv(pandas_csv_path, sep=';')

        total_time_steps = df.shape[0]
        train_data = (df.iloc[:int(total_time_steps*train_ratio),1:].values[np.newaxis, :, :], None)
        val_data = (df.iloc[int(total_time_steps*train_ratio):int(total_time_steps*(train_ratio+val_ratio)),1:].values[np.newaxis, :, :], None)
        test_data = (df.iloc[int(total_time_steps*(train_ratio+val_ratio)):,1:].values[np.newaxis, :, :], None)
    elif env_name == 'Dataset-HL':
        pandas_csv_path = './libs/datasets/data/TS_HL.csv'
        df = pd.read_csv(pandas_csv_path, sep=';')

        total_time_steps = df.shape[0]
        train_data = (df.iloc[:int(total_time_steps*train_ratio),1:].values[np.newaxis, :, :], df.iloc[:int(total_time_steps*train_ratio),:1].values[np.newaxis, :, :])
        val_data = (df.iloc[int(total_time_steps*train_ratio):int(total_time_steps*(train_ratio+val_ratio)),1:].values[np.newaxis, :, :], df.iloc[int(total_time_steps*train_ratio):int(total_time_steps*(train_ratio+val_ratio)),:1].values[np.newaxis, :, :])
        test_data = (df.iloc[int(total_time_steps*(train_ratio+val_ratio)):,1:].values[np.newaxis, :, :], df.iloc[int(total_time_steps*(train_ratio+val_ratio)):,:1].values[np.newaxis, :, :])
    elif env_name in OFFLINE_CONTROL_NPZ_FILES:
        npz_path = f"./libs/datasets/data/{OFFLINE_CONTROL_NPZ_FILES[env_name]}"
        data = np.load(npz_path)
        required_keys = {'s', 'a', 's_next', 'done'}
        missing_keys = required_keys.difference(set(data.keys()))
        if missing_keys:
            raise ValueError(f'{npz_path} is missing required keys: {sorted(missing_keys)}')

        states = data['s'].astype(np.float32)
        actions = data['a'].astype(np.float32)
        next_states = data['s_next'].astype(np.float32)
        done = data['done'].astype(np.float32).reshape(-1)

        if states.ndim != 2 or actions.ndim != 2 or next_states.ndim != 2:
            raise ValueError(f'Expected 2D arrays for s/a/s_next in {npz_path}.')
        if not (states.shape[0] == actions.shape[0] == next_states.shape[0] == done.shape[0]):
            raise ValueError(f'Inconsistent transition lengths in {npz_path}.')

        # Backward-compatible fallback to pendulum_window_length if dataset_window_length is not set.
        window_length = int(cfg_get('run.dataset_window_length', cfg_get('run.pendulum_window_length', 25)))
        state_windows, action_windows = build_transition_windows(
            states=states,
            actions=actions,
            next_states=next_states,
            done=done,
            window_length=window_length,
        )

        rng = np.random.default_rng(seed)
        permutation = rng.permutation(state_windows.shape[0])
        state_windows = state_windows[permutation]
        action_windows = action_windows[permutation]

        dataset_max_windows = int(cfg_get('run.dataset_max_windows', 0))
        if dataset_max_windows > 0:
            max_n = min(dataset_max_windows, state_windows.shape[0])
            state_windows = state_windows[:max_n]
            action_windows = action_windows[:max_n]

        num_samples = state_windows.shape[0]
        if num_samples < 3:
            raise ValueError(f'{env_name} produced only {num_samples} windows; need at least 3 for train/val/test.')

        train_end = int(num_samples * train_ratio)
        val_end = int(num_samples * (train_ratio + val_ratio))
        train_end = min(max(train_end, 1), num_samples - 2)
        val_end = min(max(val_end, train_end + 1), num_samples - 1)

        train_data = (state_windows[:train_end], action_windows[:train_end])
        val_data = (state_windows[train_end:val_end], action_windows[train_end:val_end])
        test_data = (state_windows[val_end:], action_windows[val_end:])
    else:
        raise NotImplementedError
    
    return train_data, val_data, test_data, ''

class TestEnvOptim(unittest.TestCase):
    def setUp(self):
        from hydra import initialize, compose
        initialize(config_path="../../config", version_base=None)  # Point to your actual config directory        
        self.config = compose(config_name="config.yaml")
        self.num_patients = 1000 
        load_data(config=self.config)

    def test_latest_with_pytorch_model(self):

        class StateDifferential(nn.Module):
            def __init__(self):
                super(StateDifferential, self).__init__()
                # Define the parameters for the tumor growth model
                self.alpha = nn.Parameter(torch.tensor(0.1))
                self.beta = nn.Parameter(torch.tensor(0.05))
                # Define the parameters for the chemotherapy effect
                self.gamma = nn.Parameter(torch.tensor(0.01))
                self.delta = nn.Parameter(torch.tensor(0.005))
                # Define the parameters for the radiotherapy effect
                self.epsilon = nn.Parameter(torch.tensor(0.02))
                self.zeta = nn.Parameter(torch.tensor(0.01))
                # Define a neural network for capturing complex interactions and residuals
                self.residual_nn = nn.Sequential(
                    nn.Linear(4, 10),
                    nn.ReLU(),
                    nn.Linear(10, 2)
                )

            def forward(self, tumor_volume: torch.Tensor, chemotherapy_drug_concentration: torch.Tensor, chemotherapy_dosage: torch.Tensor, radiotherapy_dosage: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                # Tumor growth model
                d_tumor_volume__dt = self.alpha * tumor_volume - self.beta * tumor_volume * chemotherapy_drug_concentration - self.epsilon * tumor_volume * radiotherapy_dosage
                # Chemotherapy drug concentration model
                d_chemotherapy_drug_concentration__dt = self.gamma * chemotherapy_dosage - self.delta * chemotherapy_drug_concentration
                # Neural network to model residuals
                residuals = self.residual_nn(torch.cat((tumor_volume.unsqueeze(1), chemotherapy_drug_concentration.unsqueeze(1), chemotherapy_dosage.unsqueeze(1), radiotherapy_dosage.unsqueeze(1)), dim=1))
                # Add residuals to the model
                d_tumor_volume__dt += residuals[:, 0]
                d_chemotherapy_drug_concentration__dt += residuals[:, 1]

                return (d_tumor_volume__dt, d_chemotherapy_drug_concentration__dt)

        train_loss, val_loss, optimized_parameters = self.env.evaluate_simulator_code_using_pytorch(StateDifferential, self.train_data, self.val_data, self.test_data, self.config)
        # train_loss, val_loss, optimized_parameters = self.env.evaluate_simulator_code_using_pytorch_with_neuroevolution(StateDifferential, self.train_data, self.val_data)
        print(f'Optimizer {self.optimizer} : Final Train MSE: {train_loss} | Final Val MSE: {val_loss}') # According to code it is 2694.2922 -- suspect data leakage error
        print(f'Optimized parameters: {optimized_parameters}')
        assert val_loss < 12.3232 * 2.0, "Val loss is too high"
        print('')



if __name__ == "__main__":
    test = TestEnvOptim()
    test.setUp()
    # test.test_latest_with_pytorch_model()
    test.test_latest_with_pytorch_model()
