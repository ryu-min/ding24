import numpy as np
import torch
import csv
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import chess
import time
import random
import os

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
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# Ваши константы и функции
MATE_VALUE_WHITE = 10000.0
MATE_VALUE_BLACK = -10000.0

def normalize_output(output):
    return -10 + (output + 10000) / 1000

class ChessDataset(Dataset):
    def __init__(self, csv_file):
        self.data = []
        with open(csv_file, mode='r') as infile:
            reader = csv.reader(infile)
            next(reader)  # пропустить заголовок

            count = 0
            for row in reader:
                count += 1
                if (count < 1010000):
                    continue
                if (count > 1040000):
                    break
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

                    self.data.append((fen, normalize_output(evaluation_value)))

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

# Загрузка модели
def load_model(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

# Предсказание и вывод результатов
def predict_and_display_results(model, dataset):
    model.eval()  # Установка модели в режим оценки
    results = []

    with torch.no_grad():
        for fen_tensor, expected_evaluation in dataset:
            output = model(fen_tensor.unsqueeze(0))  # Добавляем размерность для батча
            predicted_evaluation = output.item()  # Получаем предсказание

            # Обратная нормализация для вывода значений
            normalized_expected = expected_evaluation.item()
            results.append((fen_tensor.numpy(), normalized_expected, predicted_evaluation))

    return results

# transposition_table = {}
# def minimax_alpha_beta_cached(board, depth, model, alpha, beta, maximizing_player):
#     fen = board.fen()
    
#     if (fen, depth) in transposition_table:
#         return transposition_table[(fen, depth)]
    
#     if depth == 0 or board.is_game_over():
#         fen_tensor = fen_to_tensor(fen)
#         output = model(torch.tensor(fen_tensor.copy()).float().unsqueeze(0))
#         return output.item()

#     if maximizing_player:
#         max_eval = float('-inf')
#         for move in board.legal_moves:
#             board.push(move)
#             eval = minimax_alpha_beta_cached(board, depth - 1, model, alpha, beta, False)
#             board.pop()
#             max_eval = max(max_eval, eval)
#             alpha = max(alpha, eval)
#             if beta <= alpha:
#                 break  # Отсечение
#         transposition_table[(fen, depth)] = max_eval
#         return max_eval
#     else:
#         min_eval = float('inf')
#         for move in board.legal_moves:
#             board.push(move)
#             eval = minimax_alpha_beta_cached(board, depth - 1, model, alpha, beta, True)
#             board.pop()
#             min_eval = min(min_eval, eval)
#             beta = min(beta, eval)
#             if beta <= alpha:
#                 break  # Отсечение
#         transposition_table[(fen, depth)] = min_eval
#         return min_eval

# def predict_move(model, board, depth=3):
#     best_move = None
#     best_evaluation = float('-inf')
#     alpha = float('-inf')
#     beta = float('inf')

#     for move in board.legal_moves:
#         board.push(move)  # Применяем ход
#         evaluation = minimax_alpha_beta_cached(board, depth - 1, model, alpha, beta, False)  # Вызываем Minimax с альфа-бета
#         board.pop()  # Возвращаемся к предыдущей позиции

#         if evaluation > best_evaluation:
#             best_evaluation = evaluation
#             best_move = move

#     return best_move

transposition_table = {}

def evaluate_move(move, board):
    # Пример простой оценки: оценка основывается на материале
    board.push(move)
    evaluation = material_evaluation(board)
    board.pop()
    return evaluation

def material_evaluation(board):
    material_value = {
        'p': 1,
        'r': 5,
        'n': 3,
        'b': 3,
        'q': 9,
        'k': 0,
    }
    
    total_value = 0
    for piece in board.piece_map().values():
        total_value += material_value.get(piece.symbol(), 0) * (1 if piece.color == chess.WHITE else -1)
    
    return total_value

def minimax_alpha_beta_cached(board, depth, model, alpha, beta, maximizing_player):
    fen = board.fen()

    if (fen, depth) in transposition_table:
        return transposition_table[(fen, depth)]

    if depth == 0 or board.is_game_over():
        fen_tensor = fen_to_tensor(fen)
        output = model(torch.tensor(fen_tensor.copy()).float().unsqueeze(0))
        return output.item()

    if maximizing_player:
        max_eval = float('-inf')
        moves = sorted(board.legal_moves, key=lambda move: evaluate_move(move, board), reverse=True)  # Сортировка ходов
        for move in moves:
            board.push(move)
            eval = minimax_alpha_beta_cached(board, depth - 1, model, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Отсечение
        transposition_table[(fen, depth)] = max_eval
        return max_eval
    else:
        min_eval = float('inf')
        moves = sorted(board.legal_moves, key=lambda move: evaluate_move(move, board))  # Сортировка ходов
        for move in moves:
            board.push(move)
            eval = minimax_alpha_beta_cached(board, depth - 1, model, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Отсечение
        transposition_table[(fen, depth)] = min_eval
        return min_eval

def predict_move(model, board, depth=3):
    best_move = None
    best_evaluation = float('-inf')
    alpha = float('-inf')
    beta = float('inf')

    moves = sorted(board.legal_moves, key=lambda move: evaluate_move(move, board), reverse=True)  # Сортировка ходов
    for move in moves:
        board.push(move)  # Применяем ход
        evaluation = minimax_alpha_beta_cached(board, depth - 1, model, alpha, beta, False)  # Вызываем Minimax с альфа-бета
        board.pop()  # Возвращаемся к предыдущей позиции

        if evaluation > best_evaluation:
            best_evaluation = evaluation
            best_move = move

    return best_move

def move_to_string(move):
    """Преобразует ход в строку формата UCI (например, e2e4)."""
    return move.uci()

def clear_screen():
    """Очищает экран терминала."""
    os.system('cls' if os.name == 'nt' else 'clear')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chess AI using a neural network")
    parser.add_argument('model_file', type=str, help='Path to the saved model file')
    args = parser.parse_args()

    model = ChessCNN()  # Замените на вашу модель
    optimizer = torch.optim.Adam(model.parameters())  # Замените на ваш оптимизатор

    # Загрузка сохраненной модели
    load_model(model, optimizer, args.model_file)

    board = chess.Board()  # Инициализация шахматной доски

    # Первый ход делает нейросеть (белые)
    print("Нейросеть делает первый ход...")
    start_time = time.time()  # Начало отсчета времени
    best_move = predict_move(model, board)
    elapsed_time = time.time() - start_time  # Время выполнения
    if best_move:
        print(f"Нейросеть выбрала ход: {move_to_string(best_move)} (время: {elapsed_time:.2f} секунд)")
        board.push(best_move)

    while not board.is_game_over():
        clear_screen()  # Очистка экрана
        print(board)  # Печать доски

        # Ход игрока (черные) — выбираем случайный легальный ход
        available_moves = list(board.legal_moves)
        sample_move = random.choice(available_moves)
        print(f"Компьютер выбрал ваш ход: {move_to_string(sample_move)}")
        board.push(sample_move)

        if board.is_game_over():
            break

        # Ход нейросети (белые)
        print("Нейросеть думает над ходом...")
        start_time = time.time()  # Начало отсчета времени
        best_move = predict_move(model, board)
        elapsed_time = time.time() - start_time  # Время выполнения
        if best_move:
            # print(f"Нейросеть выбрала ход: {move_to_string(best_move)} (время: {elapsed_time:.2f} секунд)")
            board.push(best_move)

    clear_screen()  # Очистка экрана перед выводом результата
    print(board)  # Печать финальной доски

    if board.is_checkmate():
        if board.turn:  # Если это ход белых, значит черные выиграли
            print("Черные выиграли!")
        else:  # Если это ход черных, значит белые выиграли
            print("Белые выиграли!")
    elif board.is_stalemate():
        print("Ничья (партия вничью).")
    elif board.is_insufficient_material():
        print("Ничья (недостаточно материала для победы).")
    elif board.is_seventyfive_moves():
        print("Ничья (75 ходов без взятия и пешечного продвижения).")
    elif board.is_fivefold_repetition():
        print("Ничья (пятьfold повторение позиции).")

    print("Игра окончена!")
