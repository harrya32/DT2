import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import copy

class Config:
    # World parameters
    SEQ_LEN = 20  # Length of a trajectory
    N_PLANS = 3   # Number of discrete treatment plans
    STATE_DIM = 1 # Dimension of the state (e.g., blood sugar)
    
    # Data parameters
    N_TRAIN_SAMPLES = 10000
    N_TEST_SAMPLES = 1000
    BATCH_SIZE = 64
    
    # Model parameters
    EMBED_DIM = 32      # Transformer embedding dimension
    N_HEADS = 4         # Number of attention heads
    N_LAYERS = 3        # Number of transformer layers
    DROPOUT = 0.1
    
    # Training parameters
    LR = 1e-4
    EPOCHS = 25 # Set higher for better performance
    GAMMA = 0.95 # The key parameter: 0 for naive, >0 for biased
    GUMBEL_TAU = 1.0 # Temperature for Gumbel-Softmax

# --- 2. The Toy World (Ground Truth) ---
class PKPDWorld:
    """
    A simple Pharmacokinetic/Pharmacodynamic world.
    - State: blood sugar level.
    - Action: insulin dose.
    - Plans: Pre-defined sequences of insulin doses.
    """
    def __init__(self, config):
        self.config = config
        self.dt = 0.1 # time step
        # Define the 3 plans
        self.plans = {
            0: torch.tensor([10.0] + [0.0] * (config.SEQ_LEN - 1)), # Plan 0: Large dose upfront
            1: torch.tensor([5.0, 5.0] + [0.0] * (config.SEQ_LEN - 2)), # Plan 1: Split dose
            2: torch.tensor([2.0] * 5 + [0.0] * (config.SEQ_LEN - 5)), # Plan 2: Slow release
        }
        self.k1 = 0.05 # sugar decay rate
        self.k2 = 0.1  # insulin effectiveness

    def _dynamics(self, sugar, insulin_dose):
        # A simplified model of blood sugar dynamics
        glucose_intake = 2.0 * math.sin(self.dt * np.pi) # a simple meal profile
        d_sugar = (-self.k1 * sugar - self.k2 * insulin_dose + glucose_intake) * self.dt
        return sugar + d_sugar

    def simulate(self, initial_sugar, plan_id):
        if torch.is_tensor(plan_id):
            plan_id = plan_id.item()
        plan_actions = self.plans[plan_id]
        trajectory = torch.zeros(self.config.SEQ_LEN)
        current_sugar = initial_sugar.clone()
        
        for t in range(self.config.SEQ_LEN):
            current_sugar = self._dynamics(current_sugar, plan_actions[t])
            trajectory[t] = current_sugar
        return trajectory

# --- 3. The Simulator Model ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class SimulatorTransformer(nn.Module):
    """
    A Transformer model to simulate trajectories.
    Input: Initial state and a plan ID.
    Output: A predicted trajectory of states.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.plan_embedding = nn.Embedding(config.N_PLANS, config.EMBED_DIM)
        self.input_proj = nn.Linear(config.STATE_DIM, config.EMBED_DIM)
        self.pos_encoder = PositionalEncoding(config.EMBED_DIM)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.EMBED_DIM, nhead=config.N_HEADS, 
            dim_feedforward=config.EMBED_DIM * 4, dropout=config.DROPOUT, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.N_LAYERS)
        
        self.output_proj = nn.Linear(config.EMBED_DIM, config.STATE_DIM)

    def forward(self, initial_state, plan_id):
        # initial_state: [B, STATE_DIM]
        # plan_id: [B]
        
        # Project initial state and embed plan
        state_embed = self.input_proj(initial_state).unsqueeze(1) # [B, 1, D]
        plan_embed = self.plan_embedding(plan_id).unsqueeze(1) # [B, 1, D]
        
        # The first token is a combination of initial state and plan
        context_token = state_embed + plan_embed
        
        # Create dummy tokens for the rest of the sequence for the encoder to process
        dummy_tokens = torch.zeros(initial_state.size(0), self.config.SEQ_LEN - 1, self.config.EMBED_DIM, device=initial_state.device)
        
        # Full sequence for the encoder
        full_seq = torch.cat([context_token, dummy_tokens], dim=1) # [B, SEQ_LEN, D]
        full_seq = self.pos_encoder(full_seq.transpose(0,1)).transpose(0,1)
        
        # Pass through transformer
        transformer_out = self.transformer_encoder(full_seq) # [B, SEQ_LEN, D]
        
        # Project to state dimension
        predicted_trajectory = self.output_proj(transformer_out).squeeze(-1) # [B, SEQ_LEN]
        
        return predicted_trajectory

# --- 4. Helper Functions ---
def generate_data(world, n_samples):
    initial_states = torch.rand(n_samples, 1) * 10.0 # Random initial sugar levels
    
    # Behavioral policy is to choose plans randomly
    behavioral_plans = torch.randint(0, world.config.N_PLANS, (n_samples,))
    
    true_trajectories = torch.zeros(n_samples, world.config.SEQ_LEN)
    for i in range(n_samples):
        true_trajectories[i] = world.simulate(initial_states[i], behavioral_plans[i])
        
    return initial_states, behavioral_plans, true_trajectories

def value_function(trajectory):
    """
    A simple value function: lower is better.
    Penalizes high blood sugar and deviation from a target level of 5.0.
    """
    target = 5.0
    return torch.mean((trajectory - target)**2, dim=-1)

# --- 5. The Training and Evaluation Loops ---

def train(model, world, train_loader, optimizer, config, gamma):
    model.train()
    total_loss, total_sim_loss, total_dec_loss = 0, 0, 0
    
    for initial_states, _, true_trajectories in tqdm(train_loader, "Training"):
        optimizer.zero_grad()
        
        # We need to simulate outcomes for all possible plans to calculate decision loss
        batch_size = initial_states.size(0)
        
        # --- Simulation Loss Calculation (L_sim) ---
        # Simulate trajectories for all plans for the current batch's initial states
        sim_trajectories_all_plans = torch.stack([
            model(initial_states, torch.full((batch_size,), i, device=initial_states.device))
            for i in range(config.N_PLANS)
        ], dim=1) # [B, N_PLANS, SEQ_LEN]
        
        # We can't calculate a simple MSE because our dataset only has the trajectory for the *one* plan that was taken.
        # This is a key point about offline learning. For this toy example, we'll "cheat" and use the world to get true trajectories.
        # In a real scenario, you'd have to use a more complex offline loss.
        # But for proving the concept, this is fine.
        
        true_trajectories_all_plans = torch.stack([
             world.simulate(initial_states[i].squeeze(), plan_id) for i in range(batch_size) for plan_id in range(config.N_PLANS)
        ]).view(batch_size, config.N_PLANS, config.SEQ_LEN)
        
        loss_sim = F.mse_loss(sim_trajectories_all_plans, true_trajectories_all_plans)

        # --- Decision Loss Calculation (L_dec) ---
        if gamma > 0:
            # 1. Get the value of the *simulated* trajectories
            sim_values = value_function(sim_trajectories_all_plans) # [B, N_PLANS]
            
            # 2. Use Gumbel-Softmax to get a differentiable probability distribution over plans
            # We want to pick the plan with the *minimum* value, so we use negative values as logits.
            plan_probs = F.gumbel_softmax(-sim_values, tau=config.GUMBEL_TAU, hard=False) # [B, N_PLANS]
            
            # 3. Get the *true* value of each plan (again, we cheat by using the world)
            true_values = value_function(true_trajectories_all_plans) # [B, N_PLANS]
            
            # 4. The decision loss is the expected true value under the model's chosen policy.
            # We want to MINIMIZE this value.
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
    initial_states, _, _ = test_data
    n_samples = initial_states.size(0)
    
    total_mse = 0
    correct_decisions = 0
    
    with torch.no_grad():
        # Get true trajectories and values for all plans for the entire test set
        true_trajectories_all = torch.stack([
            world.simulate(initial_states[i].squeeze(), j) 
            for i in range(n_samples) for j in range(world.config.N_PLANS)
        ]).view(n_samples, world.config.N_PLANS, world.config.SEQ_LEN)
        
        true_values = value_function(true_trajectories_all)
        optimal_plan_ids = torch.argmin(true_values, dim=1)

        # Get model's predicted trajectories and values
        sim_trajectories_all = torch.stack([
            model(initial_states, torch.full((n_samples,), i, device=initial_states.device))
            for i in range(world.config.N_PLANS)
        ], dim=1)
        
        sim_values = value_function(sim_trajectories_all)
        model_plan_ids = torch.argmin(sim_values, dim=1)
        
        # Calculate metrics
        total_mse = F.mse_loss(sim_trajectories_all, true_trajectories_all).item()
        correct_decisions = (optimal_plan_ids == model_plan_ids).sum().item()
        
    return total_mse, correct_decisions / n_samples


# --- 6. Main Execution ---
if __name__ == "__main__":
    cfg = Config()
    world = PKPDWorld(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate data
    train_data = generate_data(world, cfg.N_TRAIN_SAMPLES)
    test_data = generate_data(world, cfg.N_TEST_SAMPLES)
    
    train_dataset = torch.utils.data.TensorDataset(*train_data)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    
    test_data = tuple(t.to(device) for t in test_data)

    # --- Train and Evaluate Naive Model (gamma=0) ---
    print("\n--- Training Naive Simulator (gamma = 0) ---")
    S_naive = SimulatorTransformer(cfg).to(device)
    optimizer_naive = optim.Adam(S_naive.parameters(), lr=cfg.LR)
    
    for epoch in range(cfg.EPOCHS):
        train_loss, sim_loss, dec_loss = train(S_naive, world, train_loader, optimizer_naive, cfg, gamma=0)
        print(f"Epoch {epoch+1}/{cfg.EPOCHS} | Loss: {train_loss:.4f} (Sim: {sim_loss:.4f}, Dec: {dec_loss:.4f})")
    
    print("\nEvaluating Naive Simulator...")
    mse_naive, acc_naive = evaluate(S_naive, world, test_data)
    print(f"Naive Model Results -> MSE: {mse_naive:.4f} | Decision Accuracy: {acc_naive*100:.2f}%")

    # --- Train and Evaluate Decision-Biased Model (gamma > 0) ---
    print(f"\n--- Training Decision-Biased Simulator (gamma = {cfg.GAMMA}) ---")
    S_biased = SimulatorTransformer(cfg).to(device)
    optimizer_biased = optim.Adam(S_biased.parameters(), lr=cfg.LR)

    for epoch in range(cfg.EPOCHS):
        train_loss, sim_loss, dec_loss = train(S_biased, world, train_loader, optimizer_biased, cfg, gamma=cfg.GAMMA)
        print(f"Epoch {epoch+1}/{cfg.EPOCHS} | Loss: {train_loss:.4f} (Sim: {sim_loss:.4f}, Dec: {dec_loss:.4f})")

    print("\nEvaluating Decision-Biased Simulator...")
    mse_biased, acc_biased = evaluate(S_biased, world, test_data)
    print(f"Biased Model Results -> MSE: {mse_biased:.4f} | Decision Accuracy: {acc_biased*100:.2f}%")

    # --- Final Comparison ---
    print("\n--- FINAL RESULTS ---")
    print(f"Naive Model (gamma=0):     MSE = {mse_naive:.4f}, Decision Accuracy = {acc_naive*100:.2f}%")
    print(f"Biased Model (gamma={cfg.GAMMA}):  MSE = {mse_biased:.4f}, Decision Accuracy = {acc_biased*100:.2f}%")
    
    # --- Visualize a few examples ---
    print("\nGenerating and saving visualizations...")
    S_naive.eval()
    S_biased.eval()
    
    n_viz = 4
    fig, axs = plt.subplots(n_viz, 1, figsize=(12, 3 * n_viz), sharex=True)
    fig.suptitle("Comparison of Simulator Trajectories", fontsize=16)
    initial_states_viz = test_data[0][:n_viz]

    with torch.no_grad():
        for i in range(n_viz):
            initial_state_single = initial_states_viz[i:i+1]
            
            # Get model predictions for a single plan (e.g., plan 0)
            sim_naive = S_naive(initial_state_single, torch.tensor([0], device=device)).cpu().numpy().flatten()
            sim_biased = S_biased(initial_state_single, torch.tensor([0], device=device)).cpu().numpy().flatten()
            
            # Get true trajectory for that plan
            true_traj = world.simulate(initial_state_single.squeeze(), 0).cpu().numpy().flatten()
            
            axs[i].plot(true_traj, 'k-', label='Ground Truth', linewidth=2.5, alpha=0.8)
            axs[i].plot(sim_naive, 'b--', label=f'Naive Sim')
            axs[i].plot(sim_biased, 'r:', label=f'Biased Sim')
            axs[i].set_title(f"Example {i+1} (Initial State: {initial_state_single.item():.2f})")
            axs[i].set_ylabel("State (Sugar Level)")
            axs[i].legend()
            axs[i].grid(True, linestyle='--', alpha=0.6)

    axs[-1].set_xlabel("Time Step")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to make room for suptitle
    
    # --- SAVE THE PLOT ---
    plot_filename = 'simulation_comparison_plots.png'
    plt.savefig(plot_filename, dpi=300)
    print(f"Plot saved to {plot_filename}")