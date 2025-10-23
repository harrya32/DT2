import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

# --- 1. Configuration ---
class Config:
    # World parameters
    SEQ_LEN = 20
    N_POLICIES = 3  # Changed from N_PLANS
    STATE_DIM = 1
    ACTION_DIM = 1 # Explicitly define action dimension
    
    # Data parameters
    N_TRAIN_SAMPLES = 10000
    N_TEST_SAMPLES = 1000
    BATCH_SIZE = 64
    
    # Model parameters (MLP for one-step prediction)
    HIDDEN_DIM = 128
    N_LAYERS = 3
    
    # Training parameters
    LR = 1e-4
    EPOCHS = 10 # Increased slightly for more complex task
    GAMMA = 0.01
    GUMBEL_TAU = 1.0

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
    behavioral_policies = torch.randint(0, world.config.N_POLICIES, (n_samples,))
    
    true_trajectories = torch.zeros(n_samples, world.config.SEQ_LEN)
    true_actions = torch.zeros(n_samples, world.config.SEQ_LEN)
    for i in range(n_samples):
        true_trajectories[i], true_actions[i] = world.simulate(initial_states[i], behavioral_policies[i])
        
    return initial_states, behavioral_policies, true_trajectories, true_actions

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
        current_states = next_states # The feedback loop!

    return sim_traj

# --- 5. The Training and Evaluation Loops ---
def train(model, world, train_loader, optimizer, config, gamma):
    model.train()
    total_loss, total_sim_loss, total_dec_loss = 0, 0, 0
    
    for initial_states, policy_ids, true_trajs, true_actions in tqdm(train_loader, "Training", leave=False):
        optimizer.zero_grad()
        
        # --- Simulation Loss Calculation (L_sim) ---
        # We train the one-step predictor on all steps of the sequence.
        loss_sim_steps = []
        for t in range(config.SEQ_LEN - 1):
            s_t = true_trajs[:, t].unsqueeze(1) if t == 0 else true_trajs[:, t-1].unsqueeze(1)
            a_t = true_actions[:, t].unsqueeze(1)
            s_t_plus_1_true = true_trajs[:, t].unsqueeze(1)
            
            s_t_plus_1_pred = model(s_t, a_t)
            loss_sim_steps.append(F.mse_loss(s_t_plus_1_pred, s_t_plus_1_true))
        
        loss_sim = torch.mean(torch.stack(loss_sim_steps))

        # --- Decision Loss Calculation (L_dec) ---
        if gamma > 0:
            with torch.no_grad(): # Don't need gradients for ground truth calculation
                 true_trajectories_all_policies = torch.stack([
                    world.simulate(initial_states[i], p_id)[0] 
                    for i in range(initial_states.size(0)) for p_id in range(config.N_POLICIES)
                 ]).view(initial_states.size(0), config.N_POLICIES, config.SEQ_LEN)
                 true_values = value_function(true_trajectories_all_policies)

            # Perform full autoregressive simulations for all policies to get simulated values
            sim_trajectories_all_policies = torch.stack([
                perform_autoregressive_simulation(model, world, initial_states, p_id)
                for p_id in range(config.N_POLICIES)
            ], dim=1)
            
            sim_values = value_function(sim_trajectories_all_policies)
            plan_probs = F.gumbel_softmax(-sim_values, tau=config.GUMBEL_TAU, hard=False)
            loss_dec = torch.mean(torch.sum(plan_probs * true_values, dim=-1))
        else:
            loss_dec = torch.tensor(0.0, device=initial_states.device)
            
        # --- Combine losses ---
        loss = (1 - gamma) * loss_sim + gamma * loss_dec
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_sim_loss += loss_sim.item()
        total_dec_loss += loss_dec.item()
        
    return total_loss / len(train_loader), total_sim_loss / len(train_loader), total_dec_loss / len(train_loader)


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
        
        # This MSE calculation is tricky with policies. For simplicity, we'll evaluate the full trajectory MSE.
        total_mse = F.mse_loss(sim_trajectories_all, true_trajectories_all).item()
        correct_decisions = (optimal_policy_ids == model_policy_ids).sum().item()
        
    return total_mse, correct_decisions / n_samples

# --- 6. Main Execution ---
if __name__ == "__main__":
    cfg = Config()
    world = PKPDWorld(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate and move data to device
    train_data = generate_data(world, cfg.N_TRAIN_SAMPLES)
    test_data = generate_data(world, cfg.N_TEST_SAMPLES)
    train_data_device = tuple(t.to(device) for t in train_data)
    test_data_device = tuple(t.to(device) for t in test_data)
    
    train_dataset = torch.utils.data.TensorDataset(*train_data_device)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    
    # --- Train and Evaluate Naive Model (gamma=0) ---
    print("\n--- Training Naive Autoregressive Simulator (gamma = 0) ---")
    S_naive = AutoregressiveSimulator(cfg).to(device)
    optimizer_naive = optim.Adam(S_naive.parameters(), lr=cfg.LR)
    
    for epoch in range(cfg.EPOCHS):
        train_loss, sim_loss, dec_loss = train(S_naive, world, train_loader, optimizer_naive, cfg, gamma=0)
        print(f"Epoch {epoch+1}/{cfg.EPOCHS} | Loss: {train_loss:.4f} (Sim: {sim_loss:.4f}, Dec: {dec_loss:.4f})")
    
    print("\nEvaluating Naive Simulator...")
    mse_naive, acc_naive = evaluate(S_naive, world, test_data_device)
    print(f"Naive Model Results -> MSE: {mse_naive:.4f} | Decision Accuracy: {acc_naive*100:.2f}%")

    # --- Train and Evaluate Decision-Biased Model (gamma > 0) ---
    print(f"\n--- Training Decision-Biased Autoregressive Simulator (gamma = {cfg.GAMMA}) ---")
    S_biased = AutoregressiveSimulator(cfg).to(device)
    optimizer_biased = optim.Adam(S_biased.parameters(), lr=cfg.LR)

    for epoch in range(cfg.EPOCHS):
        train_loss, sim_loss, dec_loss = train(S_biased, world, train_loader, optimizer_biased, cfg, gamma=cfg.GAMMA)
        print(f"Epoch {epoch+1}/{cfg.EPOCHS} | Loss: {train_loss:.4f} (Sim: {sim_loss:.4f}, Dec: {dec_loss:.4f})")

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
    
    n_viz = 4
    fig, axs = plt.subplots(n_viz, 1, figsize=(12, 3 * n_viz), sharex=True)
    fig.suptitle("Comparison of Autoregressive Simulator Trajectories (Policy 1 shown)", fontsize=16)
    initial_states_viz = test_data_device[0][:n_viz]

    with torch.no_grad():
        for i in range(n_viz):
            initial_state_single = initial_states_viz[i:i+1]
            
            # Perform autoregressive simulation for visualization
            sim_naive = perform_autoregressive_simulation(S_naive, world, initial_state_single, policy_id=1).cpu().numpy().flatten()
            sim_biased = perform_autoregressive_simulation(S_biased, world, initial_state_single, policy_id=1).cpu().numpy().flatten()
            
            # Get true trajectory
            true_traj = world.simulate(initial_state_single, policy_id=1)[0].cpu().numpy().flatten()
            
            axs[i].plot(true_traj, 'k-', label='Ground Truth', linewidth=2.5, alpha=0.8)
            axs[i].plot(sim_naive, 'b--', label=f'Naive Sim')
            axs[i].plot(sim_biased, 'r:', label=f'Biased Sim')
            axs[i].set_title(f"Example {i+1} (Initial State: {initial_state_single.item():.2f})")
            axs[i].set_ylabel("State (Sugar Level)")
            axs[i].legend()
            axs[i].grid(True, linestyle='--', alpha=0.6)

    axs[-1].set_xlabel("Time Step")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    plot_filename = 'autoregressive_simulation_plots.png'
    plt.savefig(plot_filename, dpi=300)
    print(f"Plot saved to {plot_filename}")
    
    plt.show()