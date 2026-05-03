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

Opening/middle game evaluation is based on the following features:
* Material difference. Evaluated using traditional piece values, except for bishops which are valued at 3.15.
* King safety. Evaluated using a FIS, independently for white and black.
* Central control. Evaluated using a FIS as a differential feature.
* Mobility. Evaluated using a FIS independently for white and black. However, the engine then utilizes the difference between white and black mobility scores for evaluation.
* Pawn structure. Same as mobility.

Endgame evaluation is based on the following features:
* Material difference. Same as in the middle game.
* Mobility. Same as in the middle game.
* Pawn structure. Same as in the middle game.
* Promotion chances. Computed and utilized in a similar way to mobility.

* **UCI Support:** includes a Universal Chess Interface (UCI) wrapper, allowing integration with standard chess GUIs (e.g. CuteChess).
* **Explanaible AI for chess:** modules to generate human-readable explanations of why a particular evaluation was given.
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

You can import FuzzyChess modules directly into your own Python scripts to evaluate specific positions or utilize the explainability features. See `notebooks/board_evaluation/board_evaluation.ipynb` for an example use case.

```python
import chess
from fuzzychess.bot.factory import load_bot
from fuzzychess.explainability.explainer import PositionExplainer
from fuzzychess.explainability.explaining_searcher import ExplainingSearcher

# Example usage
board = chess.Board()
evaluator = load_bot()
explaining_searcher = ExplainingSearcher()
explainer = PositionExplainer(evaluator, explaining_searcher)

# Get static evaluation and print and explanation
static_evaluation = evaluator.evaluate(board)
print(f"Static evaluation {static_evaluation:.2f}\n")
explainer.print_full_analysis(board)

# Alternatively, get dynamic evaluation with a simple move search and evaluate the principal variation
principal_variation, dynamic_evaluation = explaining_searcher.diagnostic_search(board, lambda board: evaluator.evaluate(board), depth = 3)

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