# Welcome to Oguz Arslan's Master Thesis Repository

## Current Description of the Project: Win Prediction GNN Model for Rocket League Analysis

This project aims to develop a deep learning model to predict the probability of a goal being scored in a Rocket League match. For any given game state, the model outputs two independent probabilities: one for the Blue team scoring and one for the Orange team scoring, both within the next **5-second window**.

The core of the project is a hybrid architecture implemented in PyTorch and PyTorch Geometric. It uses a **Graph Convolutional Network (GCN)** to model the complex, spatio-temporal relationships between all six players, and concatenates this learned representation with global game-state features (e.g., ball kinetics, game clock). This combined feature set is then processed by two separate **Multi-Layer Perceptron (MLP)** heads to generate the final predictions.

The system is trained on a large dataset of professional 3v3 match replays, with the ultimate goal of creating an interactive analytical tool in Unity for post-match analysis and "what-if" scenario exploration.

:sparkles: **Current Status:** The data processing pipeline and model architecture are fully implemented. The project is currently on the model training and evaluation phase.

### **Key Steps & Pipeline**

The project is structured into four main stages:

1. **Data Acquisition & Preprocessing:**
   * A custom Python script downloads professional match replays from ballchasing.com.
   * The carball library converts raw .replay files into structured JSON and Parquet data.
   * A robust preprocessing pipeline cleans the data, engineers new features (e.g., player forward vectors, boost pad timers), normalizes all features to a \[0, 1\] range using game-defined bounds, and generates the target labels.
   * The final output is a single, large .csv file containing millions of labeled game states.
2. **Model Architecture & Implementation:**
   * Each game state is transformed into a PyTorch Geometric Data object, containing:
     * A **fully connected player graph** where nodes are players and edge weights are the inverse distance between them.
     * A vector of **global features**.
   * The GCN processes the player graph, and a pooling layer creates a graph embedding.
   * This embedding is concatenated with the global features and fed into two MLP heads for prediction.
   * The model uses a BCEWithLogitsLoss function to effectively handle the severe class imbalance of goal events.
3. **Training & Evaluation:**
   * The model is trained on a high-performance computing cluster.
   * Performance will be evaluated using metrics suited for imbalanced classification (F1-Score, Precision, Recall, AUC-ROC).
   * Crucially, the GCN model's performance will be benchmarked against a simpler baseline (e.g., an MLP without the graph structure) to validate the core hypothesis.
4. **Analytical Interface (Prototype):**
   * A prototype interface has been developed in **Unity** to visualize replay data in a 3D environment.
   * It currently supports loading replays and modifying game state parameters.
   * **Next Step:** Integrate the trained model to enable real-time prediction updates for "what-if" analysis.
