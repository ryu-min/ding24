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

def fen_to_matrix_old(fen):
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

def fen_to_matrix(fen):
    board = chess.Board(fen)
    board_3d = np.zeros((20, 8, 8), dtype=np.float32)  # Увеличили количество каналов
    
    # Базовые фигуры (12 каналов)
    for piece in chess.PIECE_TYPES:
        for color in [chess.WHITE, chess.BLACK]:
            layer = 2*(piece-1) + (0 if color == chess.WHITE else 1)
            for square in board.pieces(piece, color):
                row, col = square_to_index(square)
                board_3d[layer][row][col] = 1

    aux = board.turn

    # Легальные ходы с классификацией (6 каналов)
    for color in [chess.WHITE, chess.BLACK]:
        board.turn = color
        for move in board.legal_moves:
            i, j = square_to_index(move.to_square)
            if board.is_capture(move):
                board_3d[12 + color][i][j] = 1  # Взятия
            elif move.promotion:
                board_3d[14 + color][i][j] = 1  # Превращения
            else:
                board_3d[16 + color][i][j] = 1  # Тихие ходы
    
        # Добавляем информацию о текущем ходе (2 канала)
    board_3d[18] = 1.0 if aux == chess.WHITE else 0.0  # Канал для белых
    board_3d[19] = 1.0 if aux == chess.BLACK else 0.0  # Канал для черных
    
    return board_3d

class ChessCNN(nn.Module):
    def __init__(self, conv_size=32, conv_depth=12, fc_depth=2):
        super(ChessCNN, self).__init__()

        # Сверточные слои
        layers = []
        in_channels = 20  # Начальное количество каналов (как в Keras)

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

class ChessCNN_New(nn.Module):
    def __init__(self, conv_size=64, conv_depth=8, fc_depth=4):
        super(ChessCNN_New, self).__init__()

        # Сверточные слои
        layers = []
        in_channels = 20  # Начальное количество каналов (14 слоев для фигур и ходов)

        for i in range(conv_depth):
            layers.append(nn.Conv2d(in_channels, conv_size, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(conv_size))
            layers.append(nn.ELU())

            # Применяем MaxPooling каждые 2 слоя, но с padding=1
            if i % 2 == 0:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

            in_channels = conv_size

        self.conv_layers = nn.Sequential(*layers)

        # Полносвязные слои
        fc_layers = []
        input_size = conv_size * 2 * 2  # Размер после свертки и подвыборки (2x2)
        
        for _ in range(fc_depth):
            fc_layers.append(nn.Linear(input_size, 128))
            fc_layers.append(nn.BatchNorm1d(128))
            fc_layers.append(nn.ELU())
            fc_layers.append(nn.Dropout(0.7))
            input_size = 128

        fc_layers.append(nn.Linear(128, 1))
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

class ChessCNN_3(nn.Module):
    def __init__(self, conv_size=32, conv_depth=4):
        super(ChessCNN_3, self).__init__()
        
        # Список для хранения сверточных слоев
        layers = []
        
        # Добавляем сверточные слои
        for _ in range(conv_depth):
            layers.append(nn.Conv2d(in_channels=18 if _ == 0 else conv_size, 
                                    out_channels=conv_size, 
                                    kernel_size=3, 
                                    padding=1))
            layers.append(nn.ReLU())
        
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.conv_layers = nn.Sequential(*layers)

        self.fc1 = nn.Linear(512, 32)  # 8x8 - размер выходного тензора после свертки
        self.fc2 = nn.Linear(32, 1)  # Выходной слой

    def forward(self, x):
        x = self.conv_layers(x)  # Проходим через сверточные слои
        x = x.view(x.size(0), -1)  # Преобразуем тензор в вектор
        x = torch.relu(self.fc1(x))  # Проходим через первый полносвязный слой с ReLU активацией
        x = torch.sigmoid(self.fc2(x))  # Выходной слой с сигмоидной активацией
        return x

class ChessCNN_4(nn.Module):
    def __init__(self, conv_size=32, conv_depth=2, fc_depth=2, dropout_rate=0.3):
        super(ChessCNN_4, self).__init__()
        
        # Сверточные слои
        conv_layers = []
        for i in range(conv_depth):
            conv_layers.append(nn.Conv2d(in_channels=14 if i == 0 else conv_size, 
                                         out_channels=conv_size, 
                                         kernel_size=8, 
                                         padding=8))
            conv_layers.append(nn.ReLU())
            # conv_layers.append(nn.MaxPool2d(kernel_size=8, stride=1))
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Полносвязные слои
        fc_layers = []
        fc_layers.append(nn.Linear(21632, 4096))  # Первый полносвязный слой
        fc_layers.append(nn.Linear(4096, 32))  # Первый полносвязный слой
        
        for _ in range(fc_depth - 1):  # Добавляем дополнительные полносвязные слои
            fc_layers.append(nn.ReLU())
            # fc_layers.append(nn.Dropout(dropout_rate))
            fc_layers.append(nn.Linear(32, 32))
        
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Linear(32, 1))  # Выходной слой
        
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv_layers(x)  # Проходим через сверточные слои
        x = x.view(x.size(0), -1)  # Преобразуем тензор в вектор
        x = self.fc_layers(x)  # Проходим через полносвязные слои
        x = torch.sigmoid(x)  # Выходной слой с сигмоидной активацией
        return x

class ChessCNN_5(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(
            # Блок 1: Анализ локальных взаимодействий
            nn.Conv2d(20, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Блок 2: Анализ диагоналей/вертикалей
            nn.Conv2d(64, 64, kernel_size=(1,3), padding=(0,1)),
            nn.Conv2d(64, 64, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Блок 3: Глобальные паттерны
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Блок 4: Комбинированный анализ
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

class ChessCNN_6(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(
            # Блок 1: Анализ локальных взаимодействий
            nn.Conv2d(20, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Блок 2: Анализ диагоналей/вертикалей
            nn.Conv2d(64, 64, kernel_size=(1,3), padding=(0,1)),
            nn.Conv2d(64, 64, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Блок 3: Глобальные паттерны
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Блок 4: Комбинированный анализ
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

class ChessCNN_7(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(
            # Блок 1: Анализ локальных взаимодействий
            nn.Conv2d(18, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Блок 2: Анализ диагоналей/вертикалей
            nn.Conv2d(64, 64, kernel_size=(1,3), padding=(0,1)),
            nn.Conv2d(64, 64, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Блок 3: Глобальные паттерны
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),
            
            # Блок 4: Комбинированный анализ
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.AdaptiveAvgPool2d((2,2))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256*2*2, 512),
            nn.ReLU(),
            nn.Dropout(0.7),
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

# def minimax_eval(board, model):
#     return model(torch.tensor(fen_to_matrix(board.fen()).copy()).float().unsqueeze(0)).item()  # Оценка хода

def minimax_eval(board, model):
    with torch.no_grad():  # Отключаем вычисление градиентов
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
    model.eval()  # Переключаем модель в режим оценки
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

    model.train()  # Возвращаем модель в режим обучения (если нужно)
    return max_move