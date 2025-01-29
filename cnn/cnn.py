import numpy as np
import chess
import torch
import torch.nn as nn

squares_index = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7,
}

def square_to_index(square):
    letter = chess.square_name(square)  # Get algebraic notation of the square
    row = 8 - int(letter[1]) # Convert rank to row index
    column = squares_index[letter[0]]  # Map file to column index using board_positions dictionary
    return row, column 

def fen_to_matrix(fen):
    board = chess.Board(fen)
    board_3d = np.zeros((14, 8, 8), dtype=np.int8)

    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.WHITE):
            index = np.unravel_index(square, (8, 8))
            board_3d[piece - 1][7 - index[0]][index[1]] = 1

        for square in board.pieces(piece, chess.BLACK):
            index = np.unravel_index(square, (8, 8))
            board_3d[piece + 5][7 - index[0]][index[1]] = 1

    aux = board.turn
    board.turn = chess.WHITE
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board_3d[12][i][j] = 1  # Layer 12

    board.turn = chess.BLACK
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board_3d[13][i][j] = 1  # Layer 13

    board.turn = aux

    return board_3d 

class ChessCNN(nn.Module):
    def __init__(self, conv_size=16, conv_depth=12, fc_depth=2):
        super(ChessCNN, self).__init__()

        # Сверточные слои
        layers = []
        in_channels = 14  # Начальное количество каналов (как в Keras)

        for _ in range(conv_depth):
            layers.append(nn.Conv2d(in_channels, conv_size, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = conv_size  # Обновляем количество входных каналов для следующего слоя

        self.conv_layers = nn.Sequential(*layers)

        # Полносвязные слои
        fc_layers = []
        input_size = conv_size * 8 * 8  # Размер после свертки, предполагается без подвыборки
        
        for _ in range(fc_depth):
            fc_layers.append(nn.Linear(input_size, 64))  # Все скрытые слои имеют 64 нейрона
            fc_layers.append(nn.ReLU())
            input_size = 64  # Обновляем размер входа для следующего слоя

        fc_layers.append(nn.Linear(64, 1))  # Один выходной нейрон
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Преобразуем тензор в вектор
        x = self.fc_layers(x)
        return x

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

    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            board.pop()
            return move 
        board.pop()

    for move in board.legal_moves:
        board.push(move)
        eval = minimax(board, depth - 1, -np.inf, np.inf, False, model)
        board.pop()
        if eval > max_eval:
            max_eval = eval
            max_move = move

    return max_move