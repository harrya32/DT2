import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

class EpisodeDataset(torch.utils.data.Dataset):
    """Dataset yielding all transitions for a single episode."""

    def __init__(self, data_tuple):
        initial_states, _, true_trajs, true_actions = data_tuple
        self.initial_states = initial_states
        self.true_trajs = true_trajs
        self.true_actions = true_actions

    def __len__(self):
        return self.initial_states.size(0)

    def __getitem__(self, idx):
        states = self.true_trajs[idx, :-1].unsqueeze(-1)
        actions = self.true_actions[idx, :-1].unsqueeze(-1)
        next_states = self.true_trajs[idx, 1:].unsqueeze(-1)
        init_state = self.initial_states[idx]
        return states, actions, next_states, init_state


# --- 1. Configuration ---
class Config:
    SEQ_LEN = 30
    N_POLICIES = 3  # Changed from N_PLANS
    STATE_DIM = 1
    ACTION_DIM = 1 # Explicitly define action dimension
    
    # Data parameters
    N_TRAIN_SAMPLES = 50000
    N_TEST_SAMPLES = 1000
    BATCH_SIZE = 64
    
    # Model parameters (MLP for one-step prediction)
    HIDDEN_DIM = 128
    N_LAYERS = 3
    
    # Training parameters
    LR = 1e-4
    EPOCHS = 100 # Increased slightly for more complex task
    GAMMA = 0.1
    GUMBEL_TAU = 1.0
    VAL_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 5
    EARLY_STOPPING_MIN_DELTA = 1e-4
    RNG_SEED = 0

# --- 2. The Toy World with Policies ---
class PKPDWorld:
    """
    World with state-dependent POLICIES instead of fixed plans.
    """
    def __init__(self, config):
        self.config = config
        self.dt = 0.1
        self.k1 = 0.05
        self.k2 = 0.1
        self.target_sugar = 5.0

    def get_action(self, policy_id, state):
        """ The core of our policies. Maps state -> action. """
        if policy_id == 0: # Aggressive Policy
            # High dose if sugar is high, otherwise nothing.
            action = 10.0 if state > self.target_sugar + 2.0 else 0.0
        elif policy_id == 1: # Reactive Policy
            # Dose proportional to how far above target we are.
            action = max(0, state - self.target_sugar) * 1.5
        elif policy_id == 2: # Cautious Policy
            # Smaller proportional dose.
            action = max(0, state - self.target_sugar) * 0.8
        else:
            action = 0.0
        return torch.tensor([action], device=state.device)

    def _dynamics(self, sugar, insulin_dose, time_step):
        glucose_intake = 2.0 * math.sin(time_step * self.dt * np.pi) 
        d_sugar = (-self.k1 * sugar - self.k2 * insulin_dose + glucose_intake) * self.dt
        return sugar + d_sugar

    def simulate(self, initial_sugar, policy_id):
        """ Simulates a trajectory using a given policy. """
        trajectory = torch.zeros(self.config.SEQ_LEN, device=initial_sugar.device)
        actions = torch.zeros(self.config.SEQ_LEN, device=initial_sugar.device)
        current_sugar = initial_sugar.clone()
        
        for t in range(self.config.SEQ_LEN):
            action = self.get_action(policy_id, current_sugar)
            current_sugar = self._dynamics(current_sugar, action, t)
            trajectory[t] = current_sugar
            actions[t] = action
        return trajectory, actions

# --- 3. The Autoregressive Simulator Model ---
class AutoregressiveSimulator(nn.Module):
    """
    Learns the one-step transition: f(s_t, a_t) -> s_{t+1}
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        layers = []
        input_dim = config.STATE_DIM + config.ACTION_DIM
        for _ in range(config.N_LAYERS):
            layers.append(nn.Linear(input_dim, config.HIDDEN_DIM))
            layers.append(nn.ReLU())
            input_dim = config.HIDDEN_DIM
        layers.append(nn.Linear(config.HIDDEN_DIM, config.STATE_DIM))
        
        self.net = nn.Sequential(*layers)

    def forward(self, state, action):
        # state: [B, STATE_DIM]
        # action: [B, ACTION_DIM]
        model_in = torch.cat([state, action], dim=-1)
        return self.net(model_in)

# --- 4. Helper and Simulation Functions ---
def generate_data(world, n_samples):
    initial_states = torch.rand(n_samples, 1) * 10.0
    #make all behavioural policies be policy 2
    behavioural_policies = torch.full((n_samples,), 2)

    true_trajectories = torch.zeros(n_samples, world.config.SEQ_LEN)
    true_actions = torch.zeros(n_samples, world.config.SEQ_LEN)
    for i in range(n_samples):
        true_trajectories[i], true_actions[i] = world.simulate(initial_states[i], behavioural_policies[i])

    return initial_states, behavioural_policies, true_trajectories, true_actions

def value_function(trajectory):
    target = 5.0
    return torch.mean((trajectory - target)**2, dim=-1)

def perform_autoregressive_simulation(model, world, initial_states, policy_id):
    """ Helper to perform the iterative simulation loop. """
    batch_size = initial_states.size(0)
    sim_traj = torch.zeros(batch_size, world.config.SEQ_LEN, device=initial_states.device)
    current_states = initial_states.clone()

    for t in range(world.config.SEQ_LEN):
        # Get action from the policy based on the *current simulated state*
        actions = torch.stack([world.get_action(policy_id, s) for s in current_states])
        # Predict the NEXT state using the model
        next_states = model(current_states, actions)
        sim_traj[:, t] = next_states.squeeze(-1)
        current_states = next_states 

    return sim_traj

# --- 5. The Training and Evaluation Loops ---
def compute_loss_terms(model, world, states, actions, next_states, init_states, config, gamma):
    pred_next = model(states, actions)
    loss_sim = F.mse_loss(pred_next, next_states)

    if gamma > 0:
        unique_init = torch.unique(init_states, dim=0)

        with torch.no_grad():
            true_trajectories_all_policies = torch.stack([
                world.simulate(unique_init[i:i+1], p_id)[0]
                for i in range(unique_init.size(0)) for p_id in range(config.N_POLICIES)
            ]).view(unique_init.size(0), config.N_POLICIES, config.SEQ_LEN)
            true_values = value_function(true_trajectories_all_policies)

        sim_trajectories_all_policies = torch.stack([
            perform_autoregressive_simulation(model, world, unique_init, p_id)
            for p_id in range(config.N_POLICIES)
        ], dim=1)

        sim_values = value_function(sim_trajectories_all_policies)
        plan_probs = F.gumbel_softmax(-sim_values, tau=config.GUMBEL_TAU, hard=False)
        loss_dec = torch.mean(torch.sum(plan_probs * true_values, dim=-1))
    else:
        loss_dec = torch.tensor(0.0, device=states.device)

    total_loss = (1 - gamma) * loss_sim + gamma * loss_dec
    return total_loss, loss_sim, loss_dec


def train_epoch(model, world, train_loader, optimizer, config, gamma, device):
    model.train()
    total_loss, total_sim_loss, total_dec_loss = 0.0, 0.0, 0.0

    for states_seq, actions_seq, next_states_seq, init_states in tqdm(train_loader, "Training", leave=False):
        states = states_seq.reshape(-1, states_seq.size(-1)).to(device)
        actions = actions_seq.reshape(-1, actions_seq.size(-1)).to(device)
        next_states = next_states_seq.reshape(-1, next_states_seq.size(-1)).to(device)
        init_states = init_states.to(device)
        if init_states.dim() == 1:
            init_states = init_states.unsqueeze(-1)

        optimizer.zero_grad()
        loss, loss_sim, loss_dec = compute_loss_terms(model, world, states, actions, next_states, init_states, config, gamma)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_sim_loss += loss_sim.item()
        total_dec_loss += loss_dec.item()

    num_batches = len(train_loader)
    return total_loss / num_batches, total_sim_loss / num_batches, total_dec_loss / num_batches


def evaluate_epoch(model, world, data_loader, config, gamma, device):
    model.eval()
    total_loss, total_sim_loss, total_dec_loss = 0.0, 0.0, 0.0

    with torch.no_grad():
        for states_seq, actions_seq, next_states_seq, init_states in data_loader:
            states = states_seq.reshape(-1, states_seq.size(-1)).to(device)
            actions = actions_seq.reshape(-1, actions_seq.size(-1)).to(device)
            next_states = next_states_seq.reshape(-1, next_states_seq.size(-1)).to(device)
            init_states = init_states.to(device)
            if init_states.dim() == 1:
                init_states = init_states.unsqueeze(-1)

            loss, loss_sim, loss_dec = compute_loss_terms(model, world, states, actions, next_states, init_states, config, gamma)
            total_loss += loss.item()
            total_sim_loss += loss_sim.item()
            total_dec_loss += loss_dec.item()

    num_batches = len(data_loader)
    return total_loss / num_batches, total_sim_loss / num_batches, total_dec_loss / num_batches


def fit_model(model, world, train_loader, val_loader, optimizer, config, gamma, device):
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    best_metrics = {}
    patience_counter = 0
    train_loss = val_loss = train_sim_loss = train_dec_loss = val_sim_loss = val_dec_loss = float("nan")

    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_sim_loss, train_dec_loss = train_epoch(model, world, train_loader, optimizer, config, gamma, device)
        val_loss, val_sim_loss, val_dec_loss = evaluate_epoch(model, world, val_loader, config, gamma, device)

        improvement = val_loss + config.EARLY_STOPPING_MIN_DELTA < best_val_loss
        if improvement:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_metrics = {
                "train_loss": train_loss,
                "train_sim_loss": train_sim_loss,
                "train_dec_loss": train_dec_loss,
                "val_loss": val_loss,
                "val_sim_loss": val_sim_loss,
                "val_dec_loss": val_dec_loss,
            }
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch}/{config.EPOCHS} | Train Loss: {train_loss:.4f} (Sim: {train_sim_loss:.4f}, Dec: {train_dec_loss:.4f}) "
            f"| Val Loss: {val_loss:.4f} (Sim: {val_sim_loss:.4f}, Dec: {val_dec_loss:.4f}) "
            f"| Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}"
        )

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    if not best_metrics:
        best_metrics = {
            "train_loss": train_loss,
            "train_sim_loss": train_sim_loss,
            "train_dec_loss": train_dec_loss,
            "val_loss": val_loss,
            "val_sim_loss": val_sim_loss,
            "val_dec_loss": val_dec_loss,
        }

    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, **best_metrics}


def evaluate(model, world, test_data):
    model.eval()
    initial_states, _, _, _ = test_data
    n_samples = initial_states.size(0)
    
    with torch.no_grad():
        # Get true values by simulating with the world
        true_trajectories_all = torch.stack([
            world.simulate(initial_states[i], j)[0]
            for i in range(n_samples) for j in range(world.config.N_POLICIES)
        ]).view(n_samples, world.config.N_POLICIES, world.config.SEQ_LEN)
        true_values = value_function(true_trajectories_all)
        optimal_policy_ids = torch.argmin(true_values, dim=1)

        # Get model's predicted values by performing autoregressive simulation
        sim_trajectories_all = torch.stack([
            perform_autoregressive_simulation(model, world, initial_states, p_id)
            for p_id in range(world.config.N_POLICIES)
        ], dim=1)
        sim_values = value_function(sim_trajectories_all)
        model_policy_ids = torch.argmin(sim_values, dim=1)
        
        # Calculate overall simulation MSE (one-step prediction error)
        all_true_s_t = true_trajectories_all[:, :, :-1].reshape(-1, 1)
        all_true_a_t = torch.stack([world.get_action(p, s) for p in range(world.config.N_POLICIES) for s in all_true_s_t]).squeeze()
        all_true_s_t1 = true_trajectories_all[:, :, 1:].reshape(-1, 1)
        
        total_mse = F.mse_loss(sim_trajectories_all, true_trajectories_all).item()
        correct_decisions = (optimal_policy_ids == model_policy_ids).sum().item()
        
    return total_mse, correct_decisions / n_samples

# --- 6. Main Execution ---
if __name__ == "__main__":
    cfg = Config()
    rng_seed = cfg.RNG_SEED
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rng_seed)

    world = PKPDWorld(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate data (stay on CPU for dataset construction)
    train_data = generate_data(world, cfg.N_TRAIN_SAMPLES)
    test_data = generate_data(world, cfg.N_TEST_SAMPLES)
    
    num_episodes = train_data[0].size(0)
    val_size = max(1, int(num_episodes * cfg.VAL_SPLIT))
    train_size = num_episodes - val_size
    if train_size <= 0:
        raise ValueError("Validation split is too large relative to the training data size.")

    generator = torch.Generator().manual_seed(rng_seed)
    perm = torch.randperm(num_episodes, generator=generator)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    def select_episode_data(data_tuple, indices):
        return tuple(t[indices] for t in data_tuple)

    train_episode_data = select_episode_data(train_data, train_idx)
    val_episode_data = select_episode_data(train_data, val_idx)

    train_dataset = EpisodeDataset(train_episode_data)
    val_dataset = EpisodeDataset(val_episode_data)

    transitions_per_episode = cfg.SEQ_LEN - 1
    episodes_per_batch = max(1, math.ceil(cfg.BATCH_SIZE / transitions_per_episode))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=episodes_per_batch, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=episodes_per_batch, shuffle=False)

    test_data_device = tuple(t.to(device) for t in test_data)
    
    # --- Train and Evaluate Naive Model (gamma=0) ---
    print("\n--- Training Naive Autoregressive Simulator (gamma = 0) ---")
    S_naive = AutoregressiveSimulator(cfg).to(device)
    optimizer_naive = optim.Adam(S_naive.parameters(), lr=cfg.LR)
    
    naive_stats = fit_model(S_naive, world, train_loader, val_loader, optimizer_naive, cfg, gamma=0, device=device)
    print(
        f"Best Naive Epoch: {naive_stats['best_epoch']} | Val Loss: {naive_stats['val_loss']:.4f} "
        f"(Sim: {naive_stats['val_sim_loss']:.4f}, Dec: {naive_stats['val_dec_loss']:.4f})"
    )
    
    print("\nEvaluating Naive Simulator...")
    mse_naive, acc_naive = evaluate(S_naive, world, test_data_device)
    print(f"Naive Model Results -> MSE: {mse_naive:.4f} | Decision Accuracy: {acc_naive*100:.2f}%")

    # --- Train and Evaluate Decision-Biased Model (gamma > 0) ---
    print(f"\n--- Training Decision-Biased Autoregressive Simulator (gamma = {cfg.GAMMA}) ---")
    S_biased = AutoregressiveSimulator(cfg).to(device)
    optimizer_biased = optim.Adam(S_biased.parameters(), lr=cfg.LR)

    biased_stats = fit_model(S_biased, world, train_loader, val_loader, optimizer_biased, cfg, gamma=cfg.GAMMA, device=device)
    print(
        f"Best Biased Epoch: {biased_stats['best_epoch']} | Val Loss: {biased_stats['val_loss']:.4f} "
        f"(Sim: {biased_stats['val_sim_loss']:.4f}, Dec: {biased_stats['val_dec_loss']:.4f})"
    )

    print("\nEvaluating Decision-Biased Simulator...")
    mse_biased, acc_biased = evaluate(S_biased, world, test_data_device)
    print(f"Biased Model Results -> MSE: {mse_biased:.4f} | Decision Accuracy: {acc_biased*100:.2f}%")

    # --- Final Comparison ---
    print("\n" + "="*40)
    print("---           FINAL RESULTS (AUTOREGRESSIVE)           ---")
    print("="*40)
    print(f"Naive Model (gamma=0):     MSE = {mse_naive:.4f}, Decision Accuracy = {acc_naive*100:.2f}%")
    print(f"Biased Model (gamma={cfg.GAMMA}):  MSE = {mse_biased:.4f}, Decision Accuracy = {acc_biased*100:.2f}%")
    print("="*40)
    
    # --- Visualize a few examples ---
    print("\nGenerating and saving visualizations...")
    S_naive.eval()
    S_biased.eval()
    
    # Visualize one example across all policies to inspect value outcomes
    fig, axs = plt.subplots(cfg.N_POLICIES, 1, figsize=(12, 3 * cfg.N_POLICIES), sharex=True)
    fig.suptitle("Single Initial State: True vs. Simulated Trajectories Across Policies", fontsize=16)
    if cfg.N_POLICIES == 1:
        axs = [axs]

    initial_state_single = test_data_device[0][0:1]
    initial_state_value = initial_state_single.squeeze().detach().cpu().item()

    with torch.no_grad():
        for policy_idx in range(cfg.N_POLICIES):
            # Generate trajectories for current policy
            true_traj = world.simulate(initial_state_single, policy_idx)[0]
            sim_naive = perform_autoregressive_simulation(S_naive, world, initial_state_single, policy_idx)
            sim_biased = perform_autoregressive_simulation(S_biased, world, initial_state_single, policy_idx)

            # Compute value function scores for display
            true_val = value_function(true_traj.unsqueeze(0)).item()
            naive_val = value_function(sim_naive).item()
            biased_val = value_function(sim_biased).item()

            true_np = true_traj.cpu().numpy().flatten()
            naive_np = sim_naive.cpu().numpy().flatten()
            biased_np = sim_biased.cpu().numpy().flatten()

            axs[policy_idx].plot(true_np, 'k-', label='Ground Truth', linewidth=2.5, alpha=0.8)
            axs[policy_idx].plot(naive_np, 'b--', label='Naive Sim')
            axs[policy_idx].plot(biased_np, 'r:', label='Biased Sim')
            axs[policy_idx].set_title(f"Policy {policy_idx} (Initial State: {initial_state_value:.2f})")
            axs[policy_idx].set_ylabel("State (Sugar Level)")
            axs[policy_idx].legend()
            axs[policy_idx].grid(True, linestyle='--', alpha=0.6)
            axs[policy_idx].text(
                0.02,
                0.95,
                f"V_true={true_val:.2f}\nV_naive={naive_val:.2f}\nV_biased={biased_val:.2f}",
                transform=axs[policy_idx].transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
            )

    axs[-1].set_xlabel("Time Step")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    plot_filename = 'autoregressive_simulation_plots.png'
    plt.savefig(plot_filename, dpi=300)
    print(f"Plot saved to {plot_filename}")
    
    plt.show()