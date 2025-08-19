# feature_importance.py (Corrected)
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import argparse

# --- 1. Copy the Model class from your training script ---
NUM_PLAYERS = 6; PLAYER_FEATURES = 13; GLOBAL_FEATURES = 14
TOTAL_FLAT_FEATURES = (NUM_PLAYERS * PLAYER_FEATURES) + GLOBAL_FEATURES

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=TOTAL_FLAT_FEATURES):
        super(BaselineMLP, self).__init__()
        self.body = nn.Sequential(nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5))
        self.orange_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.blue_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
    def forward(self, x):
        x_p = self.body(x)
        return {'orange_prediction': self.orange_head(x_p), 'blue_prediction': self.blue_head(x_p)}

# --- 2. ADD THE SKLEARN WRAPPER CLASS ---
class SklearnPyTorchWrapper:
    def __init__(self, model, team_to_predict='orange'):
        self.model = model
        self.team_to_predict = team_to_predict
        # Sklearn requires this classes_ attribute
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        # This is a dummy method. The model is already trained.
        return self

    def predict(self, X):
        # Convert numpy array to torch tensor
        X_tensor = torch.from_numpy(X).float()
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            
            # The model was trained backwards, so 'orange_prediction' is blue's goal.
            if self.team_to_predict == 'orange':
                # Use the blue head's output for orange's prediction
                probabilities = outputs['blue_prediction']
            else: # 'blue'
                # Use the orange head's output for blue's prediction
                probabilities = outputs['orange_prediction']
            
            # Apply threshold to get binary predictions
            predictions = (probabilities.numpy() > 0.5).astype(int)
        return predictions

# --- 3. Load the model and data ---
print("Loading model...")
pytorch_model = BaselineMLP()
# IMPORTANT: Make sure your 'best_model.pth' is in the same directory
checkpoint = torch.load('best_model.pth', map_location=torch.device('cpu'), weights_only=False)
pytorch_model.load_state_dict(checkpoint['model_state'])
pytorch_model.eval()

# --- Create an instance of our wrapper ---
sklearn_wrapped_model = SklearnPyTorchWrapper(pytorch_model, team_to_predict='orange')

# NOTE: This part still requires you to load your data correctly.
# The placeholder data is for demonstrating the structure.
# You need to replace this with your actual data loading and normalization.
print("Loading and preparing data (using placeholder)...")
X = np.random.rand(1000, 92).astype(np.float32) # Placeholder
y_orange = np.random.randint(0, 2, 1000)      # Placeholder

# Create a full list of your feature names IN THE CORRECT ORDER
player_feature_names = [f'p{i}_{feat}' for i in range(NUM_PLAYERS) for feat in ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'fwd_x', 'fwd_y', 'fwd_z', 'boost', 'team', 'alive', 'dist_ball']]
global_feature_names = ['ball_pos_x', 'ball_pos_y', 'ball_pos_z', 'ball_vel_x', 'ball_vel_y', 'ball_vel_z'] + [f'boost_{i}_respawn' for i in range(6)] + ['ball_hit_team', 'secs_rem']
FEATURE_NAMES = player_feature_names + global_feature_names

# --- 4. Calculate Permutation Importance ---
print("Calculating permutation importance...")
result = permutation_importance(
    estimator=sklearn_wrapped_model, # <-- Use the wrapper object here
    X=X,
    y=y_orange,
    scoring='f1', # We can now use sklearn's built-in f1 scorer
    n_repeats=10,
    random_state=42,
    n_jobs=-1 # Use all available CPU cores to speed it up
)
print("Calculation complete.")

# --- 5. Plot the results ---
sorted_idx = result.importances_mean.argsort()

# Select top 30 features to display for clarity
top_n = 30
display_idx = sorted_idx[-top_n:]

fig, ax = plt.subplots(figsize=(12, 10))
ax.boxplot(result.importances[display_idx].T, vert=False, labels=np.array(FEATURE_NAMES)[display_idx])
ax.set_title(f"Top {top_n} Permutation Importances (Orange Goal Prediction)")
ax.set_xlabel("Decrease in F1 Score")
fig.tight_layout()
plt.show()