import numpy as np
import chess
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset
import csv
import argparse
import sys

def fen_to_tensor(fen):
    board = chess.Board(fen)
    tensor = np.zeros((12, 8, 8), dtype=np.float32)  # 12 каналов для всех фигур

    piece_map = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            plane = piece_map[piece.piece_type] + (6 if piece.color == chess.BLACK else 0)
            tensor[plane, square // 8, square % 8] = 1

    is_white_turn = board.turn == chess.WHITE
    if not is_white_turn:
        tensor = np.rot90(tensor, k=2, axes=(1, 2))

    return tensor

class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()

        # Слои свертки
        self.conv_layers = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Выход (32, 4, 4)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Выход (64, 2, 2)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 2 * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


MATE_VALUE_WHITE = 10000.0
MATE_VALUE_BLACK = -10000.0

class ChessDataset(Dataset):
    def __init__(self, csv_file):
        self.data = []
        with open(csv_file, mode='r') as infile:
            reader = csv.reader(infile)
            next(reader)  # пропустить заголовок

            count = 0
            for row in reader:
                if (count > 100000):
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

                except Exception as e:
                    print(f"Unexpected error: {e} - row: {row}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen, evaluation_value = self.data[idx]

        tensor = fen_to_tensor(fen)  # Предполагается, что у вас есть функция fen_to_tensor
        tensor = torch.tensor(tensor.copy()).float()  # Преобразование в тензор PyTorch
        evaluation_tensor = torch.tensor(evaluation_value).float()

        return tensor, evaluation_tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a CSV file containing FEN strings and evaluations")
    parser.add_argument('dataset_file', type=str, help='Path to the input CSV file with training data')
    args = parser.parse_args()
    
    dataset = ChessDataset(args.dataset_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = ChessCNN()
    criterion = nn.MSELoss()  # Используйте MSE для регрессии
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    fixed_width = 50  # Ширина строки, не считая loss
    epoch_width = 6  # Ширина для epoch
    step_width = 4   # Ширина для step
    loss_width = 10   # Ширина для loss

    num_epochs = 100
    for epoch in range(num_epochs):
        step = 0        
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
    
            # Вывод прогресса
            # sys.stdout.write(f"\rEpoch [{epoch+1}/{num_epochs}], Step [{step}], Loss: {loss.item():.4f}")
            # sys.stdout.flush()

            sys.stdout.write(
                f"\rEpoch [{epoch + 1:>{epoch_width}}/{num_epochs}], "
                f"Step [{step:>{step_width}}], "
                f"Loss: {loss.item():.4f}".ljust(fixed_width)
            )
            sys.stdout.flush()
            step += 1
    
        print()  # Переход на новую строку после завершения эпохи

        # Сохранение модели и оптимизатора
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, f'model_epoch_{epoch+1}.pt')
