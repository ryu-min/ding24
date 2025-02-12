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
    letter = chess.square_name(square)
    row = 8 - int(letter[1])
    column = squares_index[letter[0]] 
    return row, column 

def fen_to_matrix(fen):
    board = chess.Board(fen)
    board_3d = np.zeros((20, 8, 8), dtype=np.float32)
    
    for piece in chess.PIECE_TYPES:
        for color in [chess.WHITE, chess.BLACK]:
            layer = 2*(piece-1) + (0 if color == chess.WHITE else 1)
            for square in board.pieces(piece, color):
                row, col = square_to_index(square)
                board_3d[layer][row][col] = 1

    aux = board.turn

    for color in [chess.WHITE, chess.BLACK]:
        board.turn = color
        for move in board.legal_moves:
            i, j = square_to_index(move.to_square)
            if board.is_capture(move):
                board_3d[12 + color][i][j] = 1
            elif move.promotion:
                board_3d[14 + color][i][j] = 1
            else:
                board_3d[16 + color][i][j] = 1
    
    board_3d[18] = 1.0 if aux == chess.WHITE else 0.0
    board_3d[19] = 1.0 if aux == chess.BLACK else 0.0
    
    return board_3d

class ChessCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(20, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 64, kernel_size=(1,3), padding=(0,1)),
            nn.Conv2d(64, 64, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2,2))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256*2*2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def load_model(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss


def minimax_eval(board, model):
    with torch.no_grad():
        return model(torch.tensor(fen_to_matrix(board.fen()).copy()).float().unsqueeze(0)).item()

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
    model.eval()
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

    model.train()
    return max_move
