import csv
import chess
import chess.engine
import chess.pgn
import argparse
import random


def extract_quiet_positions(
    pgn_path: str,
    output_csv_path: str,
    engine_path: str,
    max_positions: int = 10000,
    samples_per_game: int = 3,
    min_ply: int = 10,
    quiescence_threshold: float = 0.5,
    low_depth: int = 1,
    high_depth: int = 5,
):
    """
    Extracts quiet chess positions from a PGN file by comparing static and dynamic evaluations.
    Saves the board FEN, turn indicator, and static evaluation to a CSV file.
    """
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    saved_positions_count = 0
    game_count = 0

    print(f"Starting data extraction from {pgn_path}")

    with open(pgn_path, "r") as pgn_file, open(
        output_csv_path, "w", newline=""
    ) as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["FEN", "Turn", "Stockfish_Eval"])

        while saved_positions_count < max_positions:
            game = chess.pgn.read_game(pgn_file)
            game_count += 1
            if game is None:
                break

            all_moves = list(game.mainline_moves())
            total_plies = len(all_moves)

            # Skip games that are too short
            if total_plies <= min_ply:
                continue

            num_samples = min(samples_per_game, total_plies - min_ply)
            target_indices = set(
                random.sample(range(min_ply, total_plies), num_samples)
            )

            board = game.board()
            for idx, move in enumerate(all_moves):
                board.push(move)

                if idx in target_indices:

                    try:
                        static_analysis = engine.analyse(
                            board, chess.engine.Limit(depth=low_depth)
                        )

                        dynamic_analysis = engine.analyse(
                            board, chess.engine.Limit(depth=high_depth)
                        )

                        # Get evaluation between -10 and 10
                        static_score = (
                            static_analysis["score"].white().score(mate_score=10_000)
                            / 100.0
                        )
                        dynamic_score = (
                            dynamic_analysis["score"].white().score(mate_score=10_000)
                            / 100.0
                        )

                        # Check if the position is 'quiet' based on the threshold
                        eval_diff = abs(dynamic_score - static_score)
                        if eval_diff <= quiescence_threshold:
                            turn_indicator = board.turn

                            csv_writer.writerow(
                                [board.fen(), turn_indicator, static_score]
                            )
                            saved_positions_count += 1

                            if saved_positions_count % 500 == 0:
                                print(
                                    f"Positions saved: {saved_positions_count} / {max_positions} (Game {game_count})"
                                )

                            if saved_positions_count >= max_positions:
                                break
                    except:
                        # Skip positions that cause engine analysis erorrs
                        continue

    engine.quit()
    print(
        f"Extraction complete. {saved_positions_count} positions saved to {output_csv_path}."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract quiet positions from a .pgn file"
    )
    parser.add_argument("pgn_path", type=str, help="Path to the input .pgn")
    parser.add_argument(
        "output_csv_path", type=str, help="Path to the output .csv file"
    )
    parser.add_argument(
        "engine_path",
        type=str,
        help="Path to the chess engine executable (e.g. 'stockfish')",
    )
    parser.add_argument(
        "--max_positions",
        type=int,
        default=10_000,
        help="Maximum number of positions to extract and save in output .csv",
    )
    parser.add_argument(
        "--q_threshold",
        type=float,
        default=0.5,
        help="Quiescence threshold: maximum difference between static and dynamic "
        "evaluations for a position to be considered quiet",
    )
    parser.add_argument(
        "--samples_per_game",
        type=int,
        default=3,
        help="Maximum number of positions to analyse per game",
    )
    parser.add_argument(
        "--min_ply",
        type=int,
        default=12,
        help="Minimum number of plies in analysed positions",
    )
    args = parser.parse_args()

    extract_quiet_positions(
        pgn_path=args.pgn_path,
        output_csv_path=args.output_csv_path,
        engine_path=args.engine_path,
        max_positions=args.max_positions,
        samples_per_game=args.samples_per_game,
        min_ply=args.min_ply,
        quiescence_threshold=args.q_threshold,
        low_depth=0,
        high_depth=5,
    )


if __name__ == "__main__":
    main()
