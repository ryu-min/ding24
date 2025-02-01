from cnn import ChessCNN, fen_to_matrix
import numpy as np
import chess
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import csv
import argparse

MATE_VALUE_WHITE = 10000.0
MATE_VALUE_BLACK = -10000.0

def normalize_output(output, max_value):
    return output / max_value / 2 + 0.5

class ChessDataset(Dataset):
    def __init__(self, csv_file):
        self.data = []
        self.max_value = 0.0

        with open(csv_file, mode='r') as infile:
            reader = csv.reader(infile)
            next(reader)  # Пропустить заголовок

            count = 0
            for row in reader:
                if count > 1000000:
                    break
                count += 1
                try:
                    fen = row[0]
                    evaluation = row[1]
                    evaluation_value = 0.0

                    if evaluation.startswith('#+'):
                        evaluation_value = MATE_VALUE_WHITE
                    elif evaluation.startswith('#-'):
                        evaluation_value = MATE_VALUE_BLACK
                    else:
                        evaluation_value = float(evaluation.split()[0])

                    self.data.append((fen, evaluation_value))

                    self.max_value = max(self.max_value, abs(evaluation_value))

                except Exception as e:
                    print(f"Unexpected error: {e} - row: {row}")

        self.data = [(fen, normalize_output(evaluation_value, self.max_value)) for fen, evaluation_value in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen, evaluation_value = self.data[idx]

        tensor = fen_to_matrix(fen)
        tensor = torch.tensor(tensor.copy()).float()
        evaluation_tensor = torch.tensor(evaluation_value).float()

        return tensor, evaluation_tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a CSV file containing FEN strings and evaluations")
    parser.add_argument('dataset_file', type=str, help='Path to the input CSV file with training data')
    args = parser.parse_args()
    
    dataset = ChessDataset(args.dataset_file)
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = ChessCNN()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        step = 0        
        total_steps = len(train_dataloader)        
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            step += 1
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step}/{total_steps}], Loss: {loss.item():.12f}", end='\r', flush=True)

        average_loss = total_loss / total_steps
        print(f"\nAverage Loss for Epoch [{epoch+1}/{num_epochs}]: {average_loss:.12f}")

        # Валидация
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()

        average_val_loss = val_loss / len(val_dataloader)
        print(f"Validation Loss for Epoch [{epoch+1}/{num_epochs}]: {average_val_loss:.12f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, f'model_1kk_epoch_{epoch+1}.pt')