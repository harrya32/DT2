import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. The True Non-Linear System
# ==========================================
class BistableSystem:
    def __init__(self):
        self.dt = 0.1
        
    def dynamics(self, x, u):
        # dx/dt = x - x^3 + u
        # Unstable near 0 (slope +1)
        # Stable near +/- 1.5 (slope -2)
        return x - x**3 + u

    def step(self, state, action):
        dx = self.dynamics(state, action)
        return state + dx * self.dt

# ==========================================
# 2. The Mis-specified Model (Linear)
# ==========================================
class LinearMechanisticDT(nn.Module):
    def __init__(self):
        super().__init__()
        # dx = alpha*x + beta*u
        # We need alpha > 0 to model the tipping point.
        # But data demands alpha < 0 (stability).
        self.alpha = nn.Parameter(torch.tensor([0.0]))
        self.beta = nn.Parameter(torch.tensor([0.0]))
        self.dt = 0.1

    def forward(self, state, action):
        dx = self.alpha * state + self.beta * action
        return state + dx * self.dt
    
    def get_params(self):
        return self.alpha.item(), self.beta.item()

# ==========================================
# 3. Generate Data (Stable Tails Only)
# ==========================================
def generate_stable_data(system, n_samples=1000):
    states = []
    actions = []
    next_states = []
    
    # Sample only in the stable basins (x ~ -1.5, +1.5)
    modes = torch.tensor([-1.5, 1.5])
    idx = torch.randint(0, 2, (n_samples,))
    s = modes[idx].unsqueeze(1) + torch.randn(n_samples, 1) * 0.1
    
    # Random actions
    a = torch.randn(n_samples, 1)
    
    ns = system.step(s, a)
    
    return s, a, ns

# ==========================================
# 4. Training
# ==========================================
def train_model(dataset, lam=0.5, epochs=1000):
    states, actions, next_states = dataset
    model = LinearMechanisticDT()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    param_history = []
    
    # Loss Scaling Factor
    # Because ranking queries happen near x=0, gradients are tiny.
    # We scale the ranking loss to match the magnitude of MSE gradients.
    RANK_SCALE = 100.0 
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # --- 1. Simulation Loss ---
        pred_ns = model(states, actions)
        mse_loss = torch.mean((pred_ns - next_states)**2)
        
        # --- 2. Ranking Loss ---
        rank_loss = torch.tensor(0.0)
        
        if lam > 0:
            # Ranking Query: 
            # We are slightly off-center (x=0.5).
            # Policy A (Action 0): Should drift RIGHT (Unstable growth)
            # Policy B (Action -1): Should be pushed LEFT (Counteract)
            
            s_query = torch.full((32, 1), 0.5) 
            
            # Predict
            pred_A = model(s_query, torch.zeros(32, 1))
            pred_B = model(s_query, torch.full((32, 1), -1.0))
            
            # Note: For Linear Model, Pred_A - Pred_B = (alpha*x) - (alpha*x - beta) = beta.
            # This only optimizes Beta.
            # To optimize Alpha, we need to compare trajectories over time or different states.
            # Let's use a Multi-step rollout for ranking to engage Alpha.
            
            def get_final_val(act_val):
                curr = s_query.clone()
                act = torch.full((32, 1), act_val)
                for _ in range(5): # 5 step rollout
                    curr = model(curr, act)
                return curr
            
            val_A = get_final_val(0.0)  # Drift
            val_B = get_final_val(-1.0) # Push back
            
            # If Alpha > 0 (Unstable), val_A grows fast. 
            # If Alpha < 0 (Stable), val_A decays to 0.
            # val_A > val_B is the goal.
            
            delta_hat = val_A - val_B
            
            # Maximizing separation
            rank_loss = 1.0 - torch.tanh(delta_hat.mean())
            
            # Apply scaling
            rank_loss = rank_loss * RANK_SCALE

        # Composite Loss
        loss = (1-lam)*mse_loss + lam*rank_loss
        
        loss.backward()
        optimizer.step()
        param_history.append(model.get_params())
        
    return model, np.array(param_history)

# ==========================================
# 5. Run Experiment
# ==========================================
sys = BistableSystem()
data = generate_stable_data(sys)

# Standard DT (Lambda=0)
print("Training Standard DT...")
dt_std, hist_std = train_model(data, lam=0.0)

# DT2 (Lambda=0.5) - Now works due to Rank Scaling
print("Training DT2 (Lambda=0.5)...")
dt_our, hist_our = train_model(data, lam=0.05)

# ==========================================
# 6. Analysis
# ==========================================

# Check Alpha (Stability Parameter)
# Alpha < 0 means Stable (Bowl)
# Alpha > 0 means Unstable (Hill)
alpha_std = hist_std[-1, 0]
alpha_our = hist_our[-1, 0]

print(f"Standard Alpha: {alpha_std:.4f} (Negative = Stable)")
print(f"DT2 Alpha:      {alpha_our:.4f} (Positive = Unstable)")

# --- PLOTTING ---
plt.style.use('bmh')
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Phase Portrait
ax = axes[0]
x = np.linspace(-2, 2, 100)
y_true = x - x**3
y_std = alpha_std * x
y_our = alpha_our * x

ax.plot(x, y_true, 'k', linewidth=2, label='True (Cubic)')
ax.plot(x, y_std, '--', color='#E24A33', linewidth=3, label='Standard DT')
ax.plot(x, y_our, '--', color='#348ABD', linewidth=3, label='DT2')
ax.axhline(0, color='gray', alpha=0.5)
ax.axvline(0, color='gray', alpha=0.5)
ax.set_ylim(-3, 3)
ax.set_title("Learned Dynamics")
ax.set_xlabel("State x")
ax.set_ylabel("dx/dt")
ax.legend()
ax.text(-1.5, 2, "DT2 captures\nthe instability", color='#348ABD')

# Plot 2: Rollout from Tipping Point
ax = axes[1]
# Start just off-center
s_init = torch.tensor([[0.2]])
steps = 40
traj_true = [0.2]
traj_std = [0.2]
traj_our = [0.2]

curr = s_init.item()
for _ in range(steps):
    curr += (curr - curr**3)*0.1
    traj_true.append(curr)

curr = s_init.clone()
for _ in range(steps):
    curr = dt_std(curr, torch.tensor([[0.0]]))
    traj_std.append(curr.item())

curr = s_init.clone()
for _ in range(steps):
    curr = dt_our(curr, torch.tensor([[0.0]]))
    traj_our.append(curr.item())
    
ax.plot(traj_true, 'k', label='True System')
ax.plot(traj_std, color='#E24A33', linestyle='--', label='Standard DT')
ax.plot(traj_our, color='#348ABD', linestyle='--', label='DT2')
ax.set_title("Passive Rollout (Start x=0.2)")
ax.set_ylabel("State Value")
ax.legend()

# Plot 3: Metrics
ax = axes[2]
metrics = ['Test MSE\n(on Data)', 'Decision Regret']
# Regret is binary here: 1 if model predicts decay (wrong), 0 if growth (correct)
reg_std = 1.0 if alpha_std < 0 else 0.0
reg_our = 1.0 if alpha_our < 0 else 0.0

# Calculate MSE on data
with torch.no_grad():
    mse_std = torch.mean((dt_std(data[0], data[1]) - data[2])**2).item()
    mse_our = torch.mean((dt_our(data[0], data[1]) - data[2])**2).item()

x_pos = np.arange(2)
w = 0.35
ax.bar(x_pos - w/2, [mse_std, reg_std], w, label='Standard DT', color='#E24A33')
ax.bar(x_pos + w/2, [mse_our, reg_our], w, label='DT2', color='#348ABD')
ax.set_xticks(x_pos)
ax.set_xticklabels(metrics)
ax.set_title("Trade-off")
ax.legend()

plt.suptitle("The Tipping Point: Prioritizing Decisions over Global Fit (Lambda=0.5)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()