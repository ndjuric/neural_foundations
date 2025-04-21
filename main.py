#!/usr/bin/env python
"""
Textual User Interface for interacting with available neural network models.
"""

import os
import sys
import pkgutil
import importlib
import inspect
import requests

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

# -------------------------------------------------------------------
# Make sure `src/` is on the import path so that
#   import foundational_nn, import models_nn.*  just work.
# -------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(SCRIPT_DIR, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# =============================================================================
# Preset examples for each model
# =============================================================================
MODEL_PRESETS = {
    "Perceptron": [
        {
            "name": "OR Gate",
            "description": "Learn the OR truth table for 2 binary inputs.",
            "args": {"n_inputs": 2, "learning_rate": 0.1},
            "training_data": {
                "X": [[0, 0], [0, 1], [1, 0], [1, 1]],
                "y": [0, 1, 1, 1],
            },
        },
        {
            "name": "AND Gate",
            "description": "Learn the AND truth table for 2 binary inputs.",
            "args": {"n_inputs": 2, "learning_rate": 0.1},
            "training_data": {
                "X": [[0, 0], [0, 1], [1, 0], [1, 1]],
                "y": [0, 0, 0, 1],
            },
        },
        {
            "name": "Spam Detector (toy)",
            "description": "Toy spam detector on two features: has 'free', has 'win'.",
            "args": {"n_inputs": 2, "learning_rate": 0.2},
            "training_data": {
                "X": [[1, 1], [1, 0], [0, 1], [0, 0]],
                "y": [1, 1, 1, 0],
            },
        },
    ],
    "MarkovChain": [
        {
            "name": "Pride & Prejudice",
            "description": "Jane Austen’s classic (k=2).",
            "args": {"k": 2},
            "url": "https://www.gutenberg.org/files/1342/1342-0.txt",
        },
        {
            "name": "Sherlock Holmes",
            "description": "Arthur Conan Doyle’s adventures (k=2).",
            "args": {"k": 2},
            "url": "https://www.gutenberg.org/files/1661/1661-0.txt",
        },
        {
            "name": "Moby‑Dick",
            "description": "Herman Melville’s sea tale (k=3).",
            "args": {"k": 3},
            "url": "https://www.gutenberg.org/files/2701/2701-0.txt",
        },
    ],
    "HopfieldNetwork": [
        {
            "name": "4‑bit Patterns",
            "description": "Recall binary patterns of length 4.",
            "args": {"n_units": 4},
            "patterns": [[1, -1, 1, -1], [1, 1, -1, -1]],
        },
        {
            "name": "3×3 Letters",
            "description": "Denoise 3×3 ‘A’ & ‘O’ patterns.",
            "args": {"n_units": 9},
            "patterns": [
                [ -1, 1,-1,
                   1,-1, 1,
                   1, 1, 1 ],  # A
                [ 1, 1, 1,
                  1,-1, 1,
                  1, 1, 1 ],  # O
            ],
        },
    ],
}

# =============================================================================
# Discovery & selection of models
# =============================================================================
def list_model_classes():
    """Discover all model classes beneath src/models_nn."""
    base = os.path.join(SCRIPT_DIR, "src", "models_nn")
    if not os.path.isdir(base):
        return []
    classes = []
    for _, name, _ in pkgutil.iter_modules([base]):
        module = importlib.import_module(f"models_nn.{name}")
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ == module.__name__:
                classes.append((cls.__name__, cls))
    return classes

def select_model(classes):
    """Prompt the user to select one of the discovered models."""
    tbl = Table(title="Available Models")
    tbl.add_column("Index", style="cyan", no_wrap=True)
    tbl.add_column("Model", style="magenta")
    for idx, (name, _) in enumerate(classes):
        tbl.add_row(str(idx), name)
    console.print(tbl)
    choice = Prompt.ask("Select a model by index", choices=[str(i) for i in range(len(classes))])
    return classes[int(choice)][1]

# =============================================================================
# Helpers
# =============================================================================
def download_text(url: str) -> str:
    """Fetch text from Project Gutenberg, return None on error."""
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.text
    except Exception as e:
        console.print(f"[red]Download failed:[/] {e}")
        return None

# =============================================================================
# Demos
# =============================================================================
def run_perceptron(model, preset=None):
    from rich import box
    if preset:
        console.print(Panel(preset["description"], title=preset["name"], style="green"))
        X, y = preset["training_data"]["X"], preset["training_data"]["y"]
    else:
        X, y = [[0,0],[0,1],[1,0],[1,1]], [0,1,1,1]
    model.train(X, y, epochs=10)
    preds = [model.predict(x) for x in X]
    tbl = Table(box=box.SIMPLE)
    tbl.add_column("Input"); tbl.add_column("Predicted")
    for inp, out in zip(X, preds):
        tbl.add_row(str(inp), str(out))
    console.print(tbl)
    console.print(Panel.fit(model.visualize(), title="Perceptron Structure"))

def run_hopfield(model, preset=None):
    from rich import box
    if preset:
        console.print(Panel(preset["description"], title=preset["name"], style="green"))
        patterns = preset["patterns"]
    else:
        patterns = [[1,-1,1,-1],[1,1,-1,-1]]
    model.train(patterns)
    console.print(Panel.fit(model.visualize(), title="Hopfield Structure"))
    orig = patterns[0]
    corr = orig.copy(); corr[0] = -corr[0]
    rec  = model.recall(corr)
    tbl = Table(box=box.SIMPLE)
    tbl.add_column("State")
    tbl.add_row("Original", str(orig))
    tbl.add_row("Corrupted", str(corr))
    tbl.add_row("Recalled", str(rec))
    console.print(tbl)

def run_markov_chain(model, preset=None):
    from rich import box
    from rich.text import Text
    # 1) select text
    if preset and "url" in preset:
        console.print(Panel(preset["description"], title=preset["name"], style="green"))
        text = download_text(preset["url"])
        if not text:
            return
    else:
        text = Prompt.ask("Enter training text (blank→hello world)", default="hello world")
    # 2) preview
    preview = text[:500].strip()
    preview_text = Text(preview, overflow="fold")
    console.print(Panel(preview_text, title="Training Text Preview", height=8))
    # 3) train & gen
    model.train(text)
    length = Prompt.ask("Generated text length", default="200")
    try: length = int(length)
    except: length = 200
    seed = Prompt.ask(f"Seed (k={model.k}, optional)", default="").strip() or None
    out = model.generate(length=length, seed=seed)
    generated_text = Text(out, overflow="fold")
    console.print(Panel(generated_text, title="Generated Text", height=8))
    # 4) transitions table
    if hasattr(model, "transitions"):
        tbl = Table(title="Transition Probabilities", box=box.MINIMAL)
        tbl.add_column("State"); tbl.add_column("Next State"); tbl.add_column("Count/Prob", justify="right")
        for st, nxt in model.transitions.items():
            for k2, v in nxt.items():
                tbl.add_row(repr(st), repr(k2), str(v))
        console.print(Panel(tbl, title="Transitions", height=12))

# =============================================================================
# Main
# =============================================================================
def main():
    console.print(Text("Neural Network Explorer", style="bold underline"))
    classes = list_model_classes()
    if not classes:
        console.print("[red]No models found in src/models_nn[/]")
        sys.exit(1)

    cls = select_model(classes)
    cls_name = cls.__name__
    presets = MODEL_PRESETS.get(cls_name, [])

    if presets:
        tbl = Table(title=f"{cls_name} Examples")
        tbl.add_column("Index", style="cyan")
        tbl.add_column("Example", style="magenta")
        tbl.add_column("Description", style="green")
        for i, p in enumerate(presets):
            tbl.add_row(str(i), p["name"], p["description"])
        console.print(tbl)
        choice = Prompt.ask("Select an example", choices=[str(i) for i in range(len(presets))])
        preset = presets[int(choice)]
        model = cls(**preset.get("args", {}))

        if cls_name == "Perceptron":
            run_perceptron(model, preset)
        elif cls_name == "MarkovChain":
            run_markov_chain(model, preset)
        elif cls_name == "HopfieldNetwork":
            run_hopfield(model, preset)
        else:
            console.print(Panel.fit(model.visualize(), title=f"{cls_name} Visualization"))

    else:
        model = cls()
        if cls_name == "Perceptron":
            run_perceptron(model)
        elif cls_name == "MarkovChain":
            run_markov_chain(model)
        elif cls_name == "HopfieldNetwork":
            run_hopfield(model)
        else:
            console.print(Panel.fit(model.visualize(), title=f"{cls_name} Visualization"))

if __name__ == "__main__":
    main()