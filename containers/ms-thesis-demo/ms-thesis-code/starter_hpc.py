import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader  # Güncel DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import pandas as pd
from sklearn.model_selection import train_test_split
import wandb
import os
import argparse

# ====================== KONFİGÜRASYON ======================
NUM_PLAYERS = 6
PLAYER_FEATURES = 4  # 3D pozisyon + takım bilgisi
HIDDEN_DIM = 32

def parse_args():
    parser = argparse.ArgumentParser(description="Rocket League GCN")
    parser.add_argument('--csv-path', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()

# ====================== VERİ İŞLEME ======================
def load_and_process_data(csv_path):
    """CSV'den veri yükleme ve işleme"""
    df = pd.read_csv(csv_path)
    dataset = []
    
    for idx, row in df.iterrows():
        # Node özellikleri: [x, y, z, team]
        x = []
        for i in range(NUM_PLAYERS):
            pos = [row[f'p{i}_pos_x'], row[f'p{i}_pos_y'], row[f'p{i}_pos_z'], row[f'p{i}_team']]
            x.append(pos)
        
        x = torch.tensor(x, dtype=torch.float)
        
        # Edge hesaplama (mesafe tabanlı)
        edge_index = []
        edge_weights = []
        for i in range(NUM_PLAYERS):
            for j in range(NUM_PLAYERS):
                if i != j:
                    distance = torch.norm(x[i,:3] - x[j,:3])  # Sadece pozisyonları kullan
                    edge_index.append([i, j])
                    edge_weights.append(1.0 / (1.0 + distance))
        
        # Data objesi oluştur
        data = Data(
            x=x,
            edge_index=torch.tensor(edge_index).t().contiguous(),
            edge_weight=torch.tensor(edge_weights),
            orange_y=torch.tensor([row['team_0_goal_prev_5s']], dtype=torch.float),  # Float olarak
            blue_y=torch.tensor([row['team_1_goal_prev_5s']], dtype=torch.float)     # Float olarak
        )
        dataset.append(data)
    
    return dataset

# ====================== MODEL MİMARİSİ ======================
class RocketLeagueGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(PLAYER_FEATURES, HIDDEN_DIM)
        self.conv2 = GCNConv(HIDDEN_DIM, HIDDEN_DIM)
        
        # Sigmoid aktivasyonlu çıktı katmanları
        self.orange_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 1),
            nn.Sigmoid()
        )
        self.blue_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        
        # Grafik evrişim katmanları
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        
        # Global ortalama havuzlama
        x = global_mean_pool(x, batch)
        
        # Çıktılar [0,1] aralığında
        return self.orange_head(x), self.blue_head(x)

# ====================== EĞİTİM DÖNGÜSÜ ======================
def main():
    args = parse_args()
    
    # W&B başlatma
    wandb.login(key=os.getenv('WANDB_API_KEY'))
    wandb.init(project="rocket-league-gcn", config=args)
    
    # Veri yükleme
    dataset = load_and_process_data(args.csv_path)
    train_data, test_data = train_test_split(
        dataset, test_size=args.test_size, random_state=args.random_seed
    )
    
    # DataLoader'lar (Güncel versiyon)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    # Model ve optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RocketLeagueGCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss()

    # Eğitim döngüsü
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct_orange = 0
        correct_blue = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Model çıktıları (otomatik [0,1] aralığında)
            orange_pred, blue_pred = model(batch)
            
            # Loss hesaplama
            loss_orange = criterion(orange_pred, batch.orange_y.unsqueeze(1))
            loss_blue = criterion(blue_pred, batch.blue_y.unsqueeze(1))
            loss = loss_orange + loss_blue
            
            # Geri yayılım
            loss.backward()
            optimizer.step()
            
            # Metrikler
            total_loss += loss.item()
            correct_orange += ((orange_pred > 0.5).float() == batch.orange_y.unsqueeze(1)).sum().item()
            correct_blue += ((blue_pred > 0.5).float() == batch.blue_y.unsqueeze(1)).sum().item()
        
        # Validasyon
        model.eval()
        test_orange = test_blue = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                orange_pred, blue_pred = model(batch)
                test_orange += ((orange_pred > 0.5).float() == batch.orange_y.unsqueeze(1)).sum().item()
                test_blue += ((blue_pred > 0.5).float() == batch.blue_y.unsqueeze(1)).sum().item()
        
        # Loglama
        wandb.log({
            'epoch': epoch,
            'train_loss': total_loss/len(train_loader),
            'train_acc_orange': correct_orange/len(train_data),
            'train_acc_blue': correct_blue/len(train_data),
            'test_acc_orange': test_orange/len(test_data),
            'test_acc_blue': test_blue/len(test_data)
        })
        
        print(f"Epoch {epoch:03d} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Train Acc: O {correct_orange/len(train_data):.3f}, B {correct_blue/len(train_data):.3f} | "
              f"Test Acc: O {test_orange/len(test_data):.3f}, B {test_blue/len(test_data):.3f}")

    wandb.finish()

if __name__ == "__main__":
    main()
