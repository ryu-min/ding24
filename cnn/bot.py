import torch
import argparse
import chess
import random
import os
from cnn import ChessCNN, load_model, predict_move

def clear_screen():
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
    best_value = float('inf')
    best_moves = []

    for move in board.legal_moves:
        board.push(move)
        current_value = bot_eval_board(board)
        
        if board.is_checkmate():
            board.pop()
            return move
        
        board.pop()
        
        if current_value < best_value:
            best_value = current_value
            best_moves = [move]
        elif current_value == best_value:
            best_moves.append(move)

    if best_moves:
        return random.choice(best_moves)

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chess AI using a neural network")
    parser.add_argument('model_file', type=str, help='Path to the saved model file')
    parser.add_argument('depth', type=int, help='tree depth')
    args = parser.parse_args()

    depth = args.depth

    model = ChessCNN()
    optimizer = torch.optim.Adam(model.parameters())

    load_model(model, optimizer, args.model_file)

    board = chess.Board()

    print("Нейросеть делает первый ход...")
    best_move = predict_move(model, board, depth)
    
    if best_move:
        board.push(best_move)

    while not board.is_game_over():
        clear_screen()
        print(board)

        sample_move = choose_best_move(board)
        board.push(sample_move)

        if board.is_game_over():
            break
        
        clear_screen()
        print(board)

        best_move = predict_move(model, board, depth)
        if best_move:
            board.push(best_move)

    clear_screen()
    print(board)

    if board.is_checkmate():
        if board.turn:
            print("Черные выиграли!")
        else:
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
