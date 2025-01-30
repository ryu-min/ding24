import React, { useLayoutEffect, useState } from "react";
import { Chessboard, Square } from "react-chessboard";
import GameSession from "../../lib/session";
import { useInitialEffect } from "../../lib/utils";
import "./board.scss";
import {ShortMove} from 'chess.js';

type Props = {
  game: GameSession;
};

function parseMove(moveString: string): ShortMove {
  // Проверяем, что строка имеет длину 4
  if (moveString.length !== 4) {
      console.error('Неверный формат строки перемещения');
      throw new Error('wrong format here');
  }

  const f: string = moveString.slice(0, 2); // Первые две буквы - начальная позиция
  const t: string = moveString.slice(2, 4)
  // Создайте объект ShortMove
  const shortMove: ShortMove = { from: f as Square, to: t as Square, promotion: 'q' };

  return shortMove;
}

function Board({ game }: Props): JSX.Element {
  const [position, setPosition] = useState<string>(game.getPosition());
  const [chessboardSize, setChessboardSize] = useState<number | undefined>(
    undefined
  );

  // Observers
  useInitialEffect(() => {
    game.onBoardChange(async (position) => {
      setPosition(position); 
      if (game.getCurrentTurn() == "white") {
        console.log("first move");
        const move = await fetchBestMove(game.getPosition());
        console.log("recieve move ", move);
        if (move) {
          game.move(parseMove(move));
        } else {
          console.error("Не удалось получить лучший ход");
        }
      }
    });
  });

  // Chess resize
  useLayoutEffect(() => {
    function handleResize() {
      const display = document.getElementById("container") as HTMLElement;
      setChessboardSize(Math.min(720, display?.offsetWidth - 20));
    }

    window.addEventListener("resize", handleResize);
    handleResize();
    return () => window.removeEventListener("resize", handleResize);
  });

  const fetchBestMove = async (fen: string) => {
    try {
      const response = await fetch('http://127.0.0.1:5000/best_move', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ fen }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      return data.best_move; // Возвращаем лучший ход
    } catch (error) {
      console.error('Ошибка при получении лучшего хода:', error);
      return null;
    }
  };

  // Functions
  const onDrop = (from: Square, to: Square) => {   
    if (!game.isGameOver() && game.getCurrentTurn() == "black") {
      return !!game.move({ from, to, promotion: 'q'});
    }

    return false;
  };

  return (
    <div className="board" data-testid="board">
      <Chessboard
        animationDuration={200}
        boardOrientation="black"
        boardWidth={chessboardSize}
        position={position}
        onPieceDrop={onDrop}
        customBoardStyle={{
          borderRadius: "4px",
          boxShadow: "0 5px 15px rgba(0, 0, 0, 0.5)",
        }}
      />
    </div>
  );
}

export default Board;
