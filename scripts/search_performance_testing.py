import time
import chess
from src.bot.searcher import Searcher 
from src.bot.board_evaluation import evaluate_board

def run_benchmark():
    # 1. Start position
    # 2. Complex middlegame (Kiwipete - notorious for huge branching factors)
    # 3. Endgame position
    test_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
    ]

    search_depth = 4 # Adjust this based on how fast your Python code runs

    for i, fen in enumerate(test_fens):
        board = chess.Board(fen)
        searcher = Searcher(tt_size_mb=64)
        
        print(f"--- Position {i+1} ---")
        
        start_time = time.time()
        best_move, best_eval = searcher.alpha_beta_search(
            board=board, 
            eval_function=evaluate_board, 
            depth=search_depth
        )
        end_time = time.time()

        elapsed = end_time - start_time
        nodes = searcher.nodes_searched
        nps = int(nodes / elapsed) if elapsed > 0 else 0

        print(f"Best Move: {best_move}")
        print(f"Time:      {elapsed:.3f} seconds")
        print(f"Nodes:     {nodes:,}")
        print(f"NPS:       {nps:,} nodes/sec\n")

if __name__ == "__main__":
    run_benchmark()