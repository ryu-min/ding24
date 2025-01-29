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
from cnn import ChessCNN, fen_to_matrix

# Загрузка модели
def load_model(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

def minimax_eval(board, model):
    return model(torch.tensor(fen_to_matrix(board.fen()).copy()).float().unsqueeze(0)).item()  # Оценка хода

def minimax(board, depth, alpha, beta, maximizing_player, model):
    if depth == 0 or board.is_game_over():
        return minimax_eval(board, model)

    if maximizing_player:
        max_eval = -np.inf
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False, model)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = np.inf
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True, model)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def predict_move(model, board, depth):
    max_move = None
    max_eval = -np.inf

    # Проверяем на наличие немедленного мата
    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            board.pop()
            return move 
        board.pop()

    # Iterate over legal moves and evaluate each possible resulting board position
    for move in board.legal_moves:
        board.push(move)
        eval = minimax(board, depth - 1, -np.inf, np.inf, False, model)
        board.pop()
        # Keep track of the move with the maximum evaluation score
        if eval > max_eval:
            max_eval = eval
            max_move = move

    return max_move  # Return the AI's chosen move

def move_to_string(move):
    """Преобразует ход в строку формата UCI (например, e2e4)."""
    return move.uci()

def clear_screen():
    """Очищает экран терминала."""
    os.system('cls' if os.name == 'nt' else 'clear')

def bot_eval_board(board):
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    value = 0
    for piece_type in piece_values:
        value += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        value -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
    
    return value

def choose_best_move(board):
    best_move = None
    best_value = float('inf')  # Начальное значение для сравнения
    best_moves = []  # Список лучших ходов

    for move in board.legal_moves:
        board.push(move)  # Выполняем ход
        current_value = bot_eval_board(board)  # Оцениваем новую позицию
        
        # Проверяем, является ли ход матом
        if board.is_checkmate():
            board.pop()  # Возвращаемся к предыдущей позиции
            return move  # Возвращаем ход, который ставит мат
        
        board.pop()  # Возвращаемся к предыдущей позиции
        
        if current_value < best_value:
            best_value = current_value
            best_moves = [move]  # Сбрасываем список лучших ходов
        elif current_value == best_value:
            best_moves.append(move)  # Добавляем в список, если оценка равна

    # Если есть несколько лучших ходов, выбираем случайный
    if best_moves:
        return random.choice(best_moves)

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chess AI using a neural network")
    parser.add_argument('model_file', type=str, help='Path to the saved model file')
    parser.add_argument('depth', type=int, help='tree depth')
    args = parser.parse_args()

    depth = args.depth

    model = ChessCNN()  # Замените на вашу модель
    optimizer = torch.optim.Adam(model.parameters())  # Замените на ваш оптимизатор

    load_model(model, optimizer, args.model_file)

    board = chess.Board()

    # Первый ход делает нейросеть (белые)
    print("Нейросеть делает первый ход...")
    start_time = time.time()  # Начало отсчета времени
    best_move = predict_move(model, board, depth)
    elapsed_time = time.time() - start_time  # Время выполнения
    if best_move:
        print(f"Нейросеть выбрала ход: {move_to_string(best_move)} (время: {elapsed_time:.2f} секунд)")
        board.push(best_move)

    while not board.is_game_over():
        clear_screen()  # Очистка экрана
        print(board)  # Печать доски

        # Ход игрока (черные) — выбираем случайный легальный ход
        sample_move = choose_best_move(board)
        # print(f"Компьютер выбрал ваш ход: {move_to_string(sample_move)}")
        board.push(sample_move)

        if board.is_game_over():
            break
        
        clear_screen()  # Очистка экрана
        print(board)  # Печать доски

        # Ход нейросети (белые)
        # print("Нейросеть думает над ходом...")
        start_time = time.time()  # Начало отсчета времени
        best_move = predict_move(model, board, depth)
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
