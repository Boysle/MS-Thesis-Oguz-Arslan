import wandb
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# ====================== CONFIGURATION ======================
def parse_args():
    parser = argparse.ArgumentParser(description="Rocket League GCN")
    parser.add_argument('--csv-path', type=str, required=True, 
                        help='Path to the CSV file containing replay data')
    parser.add_argument('--batch-size', type=int, default=32, 
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30, 
                        help='Number of training epochs')
    parser.add_argument('--test-size', type=float, default=0.2, 
                        help='Test set size (0 to 1)')
    parser.add_argument('--random-seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

# ====================== MAIN EXECUTION ======================
if __name__ == "__main__":
    # Initialize W&B
    wandb.init(project="rocket-league-gcn", config=parse_args())  # Add your project name here
    config = wandb.config  # Access the config parameters passed to W&B

    # Load and process data
    dataset = load_and_process_data(config.csv_path)
    
    # Train-test split
    train_data, test_data = train_test_split(
        dataset, test_size=config.test_size, random_state=config.random_seed
    )
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config.batch_size)
    
    # Initialize model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RocketLeagueGCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    # Training loop with W&B integration
    print("\n=== TRAINING STARTED ===")
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        correct_orange = 0
        correct_blue = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            orange_pred, blue_pred = model(batch)
            
            # Calculate losses
            loss_orange = criterion(orange_pred, batch.orange_y.float().unsqueeze(1))
            loss_blue = criterion(blue_pred, batch.blue_y.float().unsqueeze(1))
            loss = loss_orange + loss_blue
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            correct_orange += ((orange_pred > 0.5) == batch.orange_y.unsqueeze(1)).sum().item()
            correct_blue += ((blue_pred > 0.5) == batch.blue_y.unsqueeze(1)).sum().item()

        # Log the loss and accuracy to W&B
        train_acc_orange = correct_orange / len(train_data)
        train_acc_blue = correct_blue / len(train_data)
        
        wandb.log({
            "epoch": epoch,
            "train_loss": total_loss/len(train_loader),
            "train_accuracy_orange": train_acc_orange,
            "train_accuracy_blue": train_acc_blue,
        })
        
        # Print epoch stats
        print(f"Epoch {epoch:02d} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Train Acc: O {train_acc_orange:.3f}, B {train_acc_blue:.3f}")
        
        # Visualizations every few epochs
        if epoch % 5 == 0:
            wandb.log({
                "model_weights": wandb.Histogram(model.state_dict())
            })

    # Final evaluation
    def evaluate(loader):
        model.eval()
        correct_orange, correct_blue = 0, 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                orange_pred, blue_pred = model(batch)
                correct_orange += ((orange_pred > 0.5) == batch.orange_y.unsqueeze(1)).sum().item()
                correct_blue += ((blue_pred > 0.5) == batch.blue_y.unsqueeze(1)).sum().item()
        return correct_orange/len(loader.dataset), correct_blue/len(loader.dataset)

    train_orange, train_blue = evaluate(train_loader)
    test_orange, test_blue = evaluate(test_loader)

    # Log final results to W&B
    wandb.log({
        "final_train_accuracy_orange": train_orange,
        "final_train_accuracy_blue": train_blue,
        "final_test_accuracy_orange": test_orange,
        "final_test_accuracy_blue": test_blue,
    })

    print("\n=== FINAL RESULTS ===")
    print(f"Train Accuracy - Orange: {train_orange:.4f}, Blue: {train_blue:.4f}")
    print(f"Test Accuracy - Orange: {test_orange:.4f}, Blue: {test_blue:.4f}")

    # Finish the W&B run
    wandb.finish()
