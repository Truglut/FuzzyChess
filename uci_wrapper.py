import chess
from src.bot.choose_move import choose_move, extract_features, EVAL_FUNCTION_REGISTRY
import sys
import argparse


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Fuzzy Chess Engine")
    parser.add_argument("--depth", type=int, default=2, help="Search depth")
    parser.add_argument(
        "--eval_type",
        type=str,
        default="trained",
        choices=["trained", "null", "material"],
        help="Determines evaluation function to be used",
    )
    parser.add_argument(
        "--quiescence", action="store_true", help="Enable quiescence search"
    )
    parser.add_argument("--name", type=str, default="FuzzyBot", help="Engine Name")

    args = parser.parse_args()

    eval_function = EVAL_FUNCTION_REGISTRY[args.eval_type]

    # Initialize the board
    board = chess.Board()

    # The infinite loop that listens to the GUI
    while True:
        try:
            line = sys.stdin.readline().strip()
            if not line:
                continue

            parts = line.split()
            command = parts[0]

            if command == "uci":
                # Tells the GUI your engine's name
                print(f"id name {args.name}")
                print("id author Andrés Contreras")
                print("uciok")

            elif command == "isready":
                print("readyok")

            elif command == "ucinewgame":
                board.reset()

            elif command == "position":
                # The GUI sends the board state. This handles both normal starts and custom FENs.
                if "startpos" in parts:
                    board.reset()
                    if "moves" in parts:
                        moves_idx = parts.index("moves")
                        for move_str in parts[moves_idx + 1 :]:
                            board.push(chess.Move.from_uci(move_str))
                elif "fen" in parts:
                    fen_start = parts.index("fen") + 1
                    if "moves" in parts:
                        moves_idx = parts.index("moves")
                        fen_str = " ".join(parts[fen_start:moves_idx])
                        board.set_fen(fen_str)
                        for move_str in parts[moves_idx + 1 :]:
                            board.push(chess.Move.from_uci(move_str))
                    else:
                        fen_str = " ".join(parts[fen_start:])
                        board.set_fen(fen_str)

            elif command == "go":
                # Choose best move
                best_move = choose_move(
                    board,
                    depth=args.depth,
                    eval_function=eval_function,
                    use_quiescence=args.quiescence,
                )

                features = extract_features(board)

                print(f"info string Características de la posición")
                print(f"info string Seguridad Rey Blanco: {features[0]:.2f}")
                print(f"info string Seguridad Rey Negro:  {features[1]:.2f}")
                print(f"info string Control Central:      {features[2]:.2f}")
                print(f"info string Material:             {features[3]:.2f}")

                # Send the chosen move back to the GUI
                print(f"bestmove {best_move.uci()}")

            elif command == "quit":
                break

            # CRITICAL: Flush the output so the GUI receives it immediately
            sys.stdout.flush()

        except Exception as e:
            # Failsafe so a crash doesn't permanently freeze the GUI
            pass


if __name__ == "__main__":
    main()
