import numpy as np
import chess
import csv
import argparse

def fen_to_tensor(fen):
    board = chess.Board(fen)
    tensor = np.zeros((8, 8), dtype=np.int32)

    piece_map = {
        chess.PAWN: 1,
        chess.KNIGHT: 2,
        chess.BISHOP: 3,
        chess.ROOK: 4,
        chess.QUEEN: 5,
        chess.KING: 6,
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            plane = piece_map[piece.piece_type] + (6 if piece.color == chess.BLACK else 0)
            tensor[square // 8, square % 8] = plane

    return tensor

def process_fen(fen, evaluation):
    tensor = fen_to_tensor(fen)
    print(f"tensor with eval {evaluation}:")
    print(tensor)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a CSV file containing FEN strings and evaluations")
    parser.add_argument('dataset_file', type=str, help='Path to the input CSV file with training data')
    args = parser.parse_args()

    input_file = args.dataset_file
    try:
        with open(input_file, mode='r') as infile:
            reader = csv.reader(infile)
            next(reader)  # skip the header
            count = 0

            for row in reader:
                if count >= 100:
                    break
                fen = row[0]
                evaluation = row[1]
                process_fen(fen, evaluation)
                count += 1
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred while opening the file: {e}")
