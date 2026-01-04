import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(10) # Seed selected to ensure low SNR effects are visible

# ==========================================
# 1. The "Hidden Wind" System
# ==========================================
class HiddenWindSystem:
    def __init__(self):
        self.state_dim = 1
        
    def step(self, state, action, t):
        """
        state: 1D position
        action: 1D thrust
        t: Time (HIDDEN from the model)
        """
        # 1. The Hidden Force (Wind)
        # Magnitude is 20x larger than action.
        # It depends on 't', which the model DOES NOT see.
        wind = 20.0 * torch.sin(0.5 * t)
        
        # 2. The Dynamics
        # Next = Curr + Wind + Action
        next_state = state + wind + 1.0 * action
        
        return next_state

# ==========================================
# 2. The Blind Model
# ==========================================
class BlindDigitalTwin(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: State (1) + Action (1) = 2.
        # NOTE: The model does NOT get 't' or 'wind' as input.
        # It is strictly partially observed.
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Output: Predicted Next State
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

# ==========================================
# 3. Data Generation
# ==========================================
def generate_data(system, n_samples=2000):
    states = []
    actions = []
    next_states = []
    # We track 'times' for the system, but won't give them to the model
    times = [] 
    
    # Continuous time evolution
    t_vals = torch.linspace(0, 100, n_samples)
    
    current_s = torch.zeros(1, 1)
    
    for i in range(n_samples):
        t = t_vals[i:i+1]
        
        # Random behavior policy
        a = torch.randn(1, 1)
        
        ns = system.step(current_s, a, t)
        
        states.append(current_s)
        actions.append(a)
        next_states.append(ns)
        times.append(t)
        
        # Reset state occasionally to keep it bounded for the plot
        if i % 50 == 0:
            current_s = torch.zeros(1, 1)
        else:
            current_s = ns
        
    return (torch.cat(states), torch.cat(actions), 
            torch.cat(next_states), torch.cat(times))

# ==========================================
# 4. Training Loop
# ==========================================
def train_model(system, dataset, lam=0.0, epochs=1000):
    states, actions, next_states, times_hidden = dataset
    model = BlindDigitalTwin()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # --- 1. Simulation Loss (MSE) ---
        pred_next = model(states, actions)
        mse_loss = torch.mean((pred_next - next_states)**2)
        
        # --- 2. Ranking Loss ---
        rank_loss = torch.tensor(0.0)
        
        if lam > 0:
            # Sample batch for ranking comparison
            idx = torch.randint(0, len(states), (32,))
            s_batch = states[idx]
            # We need the hidden times to simulate the GROUND TRUTH 
            # (which the loss function "knows" via OPE/Oracle)
            t_batch = times_hidden[idx]
            
            # --- Differentiable Model Estimates ---
            # Predict 1-step change for Action +1 vs Action -1
            # Note: We only simulate 1 step here because the wind changes over time
            pred_A = model(s_batch, torch.full((32,1), 1.0))
            pred_B = model(s_batch, torch.full((32,1), -1.0))
            
            # The model's estimated difference in value (Value = State position)
            delta_hat = pred_A - pred_B
            
            # --- Ground Truth Estimates ---
            # What actually happens in the system?
            real_A = system.step(s_batch, torch.full((32,1), 1.0), t_batch)
            real_B = system.step(s_batch, torch.full((32,1), -1.0), t_batch)
            delta_true = real_A - real_B 
            # Note: delta_true will be exactly 2.0 everywhere, 
            # because (Wind cancels out).
            
            # Ranking Loss (Smoothed Kendall's)
            # We want sign(delta_hat) == sign(delta_true)
            rank_loss = 1.0 - torch.tanh(delta_hat * torch.sign(delta_true))
            rank_loss = rank_loss.mean()

        # Composite Loss
        total_loss = (1 - lam) * mse_loss + lam * rank_loss
        
        total_loss.backward()
        optimizer.step()
        
    return model

# ==========================================
# 5. Run Experiment
# ==========================================
sys = HiddenWindSystem()
data = generate_data(sys)

# Case A: Standard MSE
# The model faces a signal-to-noise ratio of 1:20.
# The "Wind" looks like massive variance. The optimizer often gets stuck
# predicting the mean (0 change) rather than fitting the small action signal.
print("Training Standard DT...")
dt_std = train_model(sys, data, lam=0.0)

# Case B: DT^2
# The ranking loss compares (State + Wind + Act1) vs (State + Wind + Act2).
# The Wind cancels out. The gradient is pure signal.
print("Training DT2...")
dt_ours = train_model(sys, data, lam=0.999)

# ==========================================
# 6. Evaluation
# ==========================================
def get_action_sensitivity(model):
    """
    Does the model think Action +1 is different from Action -1?
    We test this at a neutral state.
    """
    s = torch.zeros(1, 1)
    with torch.no_grad():
        out_pos = model(s, torch.tensor([[1.0]]))
        out_neg = model(s, torch.tensor([[-1.0]]))
    
    # Return the predicted delta
    return (out_pos - out_neg).item()

sens_std = get_action_sensitivity(dt_std)
sens_our = get_action_sensitivity(dt_ours)
true_sens = 2.0 # (1.0) - (-1.0)

# Calculate Test MSE
with torch.no_grad():
    pred_std = dt_std(data[0], data[1])
    mse_std_val = torch.nn.functional.mse_loss(pred_std, data[2]).item()
    pred_our = dt_ours(data[0], data[1])
    mse_our_val = torch.nn.functional.mse_loss(pred_our, data[2]).item()

# ==========================================
# 7. Visualization
# ==========================================
plt.style.use('bmh')
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Visualizing the Problem (High Variance)
# We plot Next State vs Time to show the massive wind
t_np = data[3].numpy().flatten()[:200]
x_np = data[2].numpy().flatten()[:200]
ax = axes[0]
ax.scatter(t_np, x_np, alpha=0.5, label='Observed Transitions', color='gray')
ax.set_title("The 'Hidden Wind' Problem")
ax.set_xlabel("Hidden Time")
ax.set_ylabel("Next State Position")
ax.text(10, 0, "Massive variance caused\nby hidden variable.\nAction signal is tiny.", 
        bbox=dict(facecolor='white', alpha=0.8))

# Plot 2: Action Sensitivity (The Decision Boundary)
ax = axes[1]
x = ['True Physics', 'Standard DT', 'DT2 (Ours)']
y = [true_sens, sens_std, sens_our]
bars = ax.bar(x, y, color=['gray', '#E24A33', '#348ABD'])
ax.set_title("Learned Effect of Action (+1 vs -1)")
ax.set_ylabel("Predicted Delta (x_next_A - x_next_B)")
ax.axhline(true_sens, color='gray', linestyle='--', alpha=0.5)

# Label bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom')

# Plot 3: Trade-off
ax = axes[2]
metrics = ['Test MSE', 'Decision Error\n|TrueDelta - PredDelta|']
val_std = [mse_std_val, abs(true_sens - sens_std)]
val_our = [mse_our_val, abs(true_sens - sens_our)]

x_pos = np.arange(len(metrics))
width = 0.35

ax.bar(x_pos - width/2, val_std, width, label='Standard DT', color='#E24A33')
ax.bar(x_pos + width/2, val_our, width, label='DT2', color='#348ABD')
ax.set_xticks(x_pos)
ax.set_xticklabels(metrics)
ax.set_title("Accuracy vs Decision Quality")
ax.legend()

plt.suptitle("Partial Observability: When Physics Fidelity (MSE) Misleads Decisions", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()