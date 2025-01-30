from flask import Flask, request, jsonify
import chess
import torch
from cnn import ChessCNN, load_model, predict_move
import argparse
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Определяем функцию для загрузки модели
def load_chess_model(model_path):
    model = ChessCNN()
    optimizer = torch.optim.Adam(model.parameters())
    load_model(model, optimizer, model_path)
    return model

@app.route('/best_move', methods=['POST'])
def get_best_move():
    data = request.json
    fen = data.get('fen')
    
    if not fen:
        return jsonify({'error': 'FEN string is required'}), 400

    board = chess.Board(fen)
    best_move = predict_move(model, board, depth)

    if best_move:
        return jsonify({'best_move': best_move.uci()})
    else:
        return jsonify({'error': 'No legal moves available'}), 400

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Запуск REST-сервиса для шахматного AI")
    parser.add_argument('model_file', type=str, help='Path to the saved model file')
    parser.add_argument('--depth', type=int, default=3, help='Depth for move prediction')
    
    args = parser.parse_args()

    # Загружаем модель с указанного пути
    model = load_chess_model(args.model_file)
    depth = args.depth

    app.run(debug=True)
