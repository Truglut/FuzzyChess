# FuzzyChess

> A chess engine and evaluator based on fuzzy logic.

FuzzyChess is a chess engine that utilizes Fuzzy Inference Systems (FIS) to evaluate chess positions.

It implements a hierarchical structure, where a series of (Mamdani-type) FIS evaluate human-readable features of the board such as *king safety*, *central control*, *piece mobility*, etc. A second FIS (order zero Sugeno) then evaluates the position using said features as well as some other non-fuzzy features like material counts.

**Play against FuzzyChess:** the engine is connected to Lichess as a bot at @FuzzyChessBot.

*Note: as this is a personal project, the bot is currently hosted locally and does not have 24/7 uptime. If you see it online, feel free to challenge it! You can also review its past games and performance on its profile.*

## Overview

This repository contains the source code for the FuzzyChess engine, the evaluation logic, data extraction scripts, and Jupyter notebooks used for designing the fuzzy systems. It also includes an explainability module designed to interpret and describe the reasoning behind the engine's positional evaluations.

### Key Features

* **Fuzzy Evaluation Engine:** evaluates positions based on a hierarchical fuzzy inference system. The engine utilizes tapered evaluation to combine potentially different opening/middle game and endgame evaluations.

The features each evaluation is based are described in the following table:

| Feature                 | Opening / Middle game                                                                              | Endgame                                      |
| :---------------------- | :------------------------------------------------------------------------------------------------- | :------------------------------------------- |
| **Material difference** | Evaluated using traditional piece values (bishops = 3.15).                                        | Same as middle game.                         |
| **King Safety**         | Evaluated using a FIS, independently for white and black.                                          | Not evaluated.                               |
| **Central Control**     | Evaluated using a FIS as a differential feature.                                                   | Not evaluated.                               |
| **Mobility**            | Evaluate using a FIS independently. The engine uses the difference between white and black scores. | Same as middle game.                         |
| **Pawn structure**      | Evaluated and utilized similarly to mobility.                                                      | Same as middle game.                         |
| **Promotion chanches**  | Not evaluated.                                                                                     | Evaluated and utilized similarly to mobility. |

* **UCI Support:** includes a Universal Chess Interface (UCI) wrapper, allowing integration with standard chess GUIs (e.g. CuteChess).
* **Explainable AI for chess:** modules to generate human-readable explanations of why a particular evaluation was given.
* **(Anti-)Symmetric evaluation:** the evaluation system is built to ensure it treats white and black pieces equally.


## Repository Structure

The project is organized as follows:

* `data/`: contains lookup tables (LUTs), trained models, and extracted position datasets.
* `notebooks/`: Jupyter notebooks detailing the design and tuning of the Fuzzy Inference Systems.
* `scripts/`: utilities for data extraction, experiment execution, search performance testing, and model training.
* `src/fuzzychess/`: the core package containing the engine, evaluation logic, explainability tools, and UCI interface.
* `tests/`: unit and integration tests for the various components.


## Installation

### Prerequisites

* Python 3.10+

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Truglut/FuzzyChess.git
    cd FuzzyChess
```

2. It is recommended to use a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

3. Install the package and its dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### As a UCI Engine

To use FuzzyChess in a standard chess GUI, configure the GUI to point to the entry script to the engine (`src/fuzzychess/interface/uci_wrapper.py`).

For GUIs that require an executable file, a solution is to create a shell script or `.bat` file that runs the `uci_wrapper.py` script. You can set the bot's search depth in the shell script/batch file.

```bash
.venv/bin/python -u src/fuzzychess/interface/uci_wrapper.py --depth 2
```

### As a Python Library

You can import FuzzyChess modules directly into your own Python scripts to evaluate specific positions or utilize the explainability features. An example of its usage can be found in the following code block. See `notebooks/board_evaluation/board_evaluation.ipynb` for a more detailed example use case.

```python
import chess
from fuzzychess.bot.factory import load_bot
from fuzzychess.explainability.explainer import PositionExplainer
from fuzzychess.explainability.explaining_searcher import ExplainingSearcher

# Example usage

# Initialize the chess.Board object using a FEN string
example_position = "r1b2rk1/ppp2pp1/3p3p/3Pn3/4N3/2PBP3/P1P3PP/1R3RK1 w - - 1 20"
board = chess.Board(example_position)

# Load the evaluator, searcher and explainer
evaluator = load_bot()
explaining_searcher = ExplainingSearcher()
explainer = PositionExplainer(evaluator, explaining_searcher)

# Get static evaluation and print and explanation
static_evaluation = evaluator.evaluate(board)
print(f"Static evaluation {static_evaluation:.2f}\n")
explainer.print_full_analysis(board)

# Alternatively, get dynamic evaluation with a simple move search and evaluate the principal variation
principal_variation, dynamic_evaluation = explaining_searcher.diagnostic_search(
    board,
    lambda board: evaluator.evaluate(board),
    depth=3
)

# Print principal variation
print("---- Principal Variation ----")
for i, move in enumerate(principal_variation):
    print(f"{i + 1}. {board.san(move)}")
    board.push(move)

# Get explanation for the evaluated position
explainer.print_full_analysis(board)

# Remember to board.pop() the moves if needed
```

Author: Andrés Contreras Santos