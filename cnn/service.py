from flask import Flask, request, jsonify
import chess
import torch
from cnn import ChessCNN, ChessCNN_3, ChessCNN_4, ChessCNN_5, ChessCNN_6, ChessCNN_7, load_model, predict_move
import argparse
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def load_chess_model(model_path):
    model = ChessCNN_6()
    optimizer = torch.optim.Adam(model.parameters())
    load_model(model, optimizer, model_path)
    return model

@app.route('/best_move', methods=['POST'])
def get_best_move():
    data = request.json
    fen = data.get('fen')
    print(fen)

    if not fen:
        return jsonify({'error': 'FEN string is required'}), 400

    if fen == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1":
        return jsonify({'best_move': "e2e4"})

    board = chess.Board(fen)
    best_move = predict_move(model, board, depth)

    if best_move:
        print(f"move is {best_move.uci()}")
        return jsonify({'best_move': best_move.uci()})
    else:
        return jsonify({'error': 'No legal moves available'}), 400

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Запуск REST-сервиса для шахматного AI")
    parser.add_argument('model_file', type=str, help='Path to the saved model file')
    args = parser.parse_args()
    # model_file =  ".\model_random_new_6_epoch_3.pt"
    model_file = args.model_file
    depth = 2
    
    model = load_chess_model(model_file)

    app.run(debug=True)
