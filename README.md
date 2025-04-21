# Foundational Neural Network Library

[![Python 3.7+](https://img.shields.io/badge/python-3.7%2B-blue)]()

> > This project is a deliberate reconstruction of neural network architectures from their fundamental primitives, inspired by the Asimov Institute's "[Neural Network Zoo](https://www.asimovinstitute.org/neural-network-zoo/)" visualization. The intent is pedagogical: to rigorously explore the first principles of these models through constructive implementation. While leveraging performant numerical libraries (NumPy, Pandas), the architecture emphasizes modularity and explicit composition. Examine the codebase to observe this structure directly; contributions and critical analysis are welcomed. Clone to inspect the underlying mechanics and extend as desired.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Interactive Demo](#interactive-demo)
  - [Programmatic Usage](#programmatic-usage)
- [Examples](#examples)
- [Project Structure](#project-structure)
- [Extending the Library](#extending-the-library)
- [Contributing](#contributing)
- [License](#license)

## Overview

This library provides:

- **Atomic Primitives** (`foundational_nn`): Base cells, layers, activation functions, and modifiers.
- **Modular Architectures** (`architecture_nn`): Dense, convolutional, recurrent, pooling layers, `SequentialModel`, `GraphModel`, and common patterns.
- **Ready-to-Use Models** (`models_nn`): Perceptron (logic gates), Hopfield Network (associative memory), Markov Chain (text generation).
- **Interactive TUI**: A Rich-based terminal interface for guided experimentation and visualization.

All implementations emphasize clarity, extensibility, and educational value.

## Features

- Build and visualize custom neural network architectures.
- Train and test basic models: logic gates, associative memory, text generation.
- Export network diagrams in [Mermaid](https://mermaid-js.github.io) syntax.
- Clean, decoupled design 

## Getting Started

### Installation

```bash
git clone https://github.com/ndjuric/neural_foundations.git
cd <repo>
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .\.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

### Interactive Demo

Launch the interactive Textual UI:

```bash
python3 main.py
```

Follow the on-screen prompts to select a model and preset example.

### Programmatic Usage

Use the library in your own Python scripts:

```python
from models_nn.perceptron import Perceptron

# Create a Perceptron for OR logic
p = Perceptron(n_inputs=2, learning_rate=0.1)
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 1]
p.train(X, y, epochs=10)
print([p.predict(x) for x in X])
print(p.visualize())  # Mermaid diagram, WIP
```

## Examples

1. **Perceptron OR Gate**
   ```bash
   python3 main.py          # Select "Perceptron" → "OR Gate"
   ```

2. **Hopfield Network Associative Memory**
   ```bash
   python3 main.py          # Select "HopfieldNetwork" → "4-bit Patterns"
   ```

3. **Markov Chain Text Generation**
   ```bash
   python3 main.py          # Select "MarkovChain" → "Lorem Ipsum"
   ```

## Project Structure

```text
.
├── main.py                # Interactive TUI entrypoint
├── requirements.txt       # Project dependencies
├── README.md              # This file
└── src/
    ├── foundational_nn    # Base primitives: cells, activations, modifiers
    ├── architecture_nn    # Layers, models, patterns
    └── models_nn          # Ready-to-use example models
```

## Extending the Library

1. Define new primitives in `foundational_nn`.
2. Create custom layers or network patterns in `architecture_nn`.
3. Package new models with demos in `models_nn`.

## Contributing

Contributions are welcome! Please open issues and submit pull requests.

## License

Licensed under the MIT License.
