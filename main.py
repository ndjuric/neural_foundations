#!/usr/bin/env python
"""
Textual User Interface for interacting with available neural network models.
"""
import os
import sys
import pkgutil
import importlib
import inspect

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Ensure 'src' directory is on the import path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(SCRIPT_DIR, 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

console = Console()

# Predefined example presets for each model
MODEL_PRESETS = {
    'Perceptron': [
        {
            'name': 'OR Gate',
            'description': 'Learn the OR truth table for 2 binary inputs.',
            'args': {'n_inputs': 2, 'learning_rate': 0.1},
            'training_data': {
                'X': [[0, 0], [0, 1], [1, 0], [1, 1]],
                'y': [0, 1, 1, 1],
            },
        },
        {
            'name': 'AND Gate',
            'description': 'Learn the AND truth table for 2 binary inputs.',
            'args': {'n_inputs': 2, 'learning_rate': 0.1},
            'training_data': {
                'X': [[0, 0], [0, 1], [1, 0], [1, 1]],
                'y': [0, 0, 0, 1],
            },
        },
        {
            'name': 'XOR Gate',
            'description': 'Attempt to learn the XOR truth table (should fail).',
            'args': {'n_inputs': 2, 'learning_rate': 0.1},
            'training_data': {
                'X': [[0, 0], [0, 1], [1, 0], [1, 1]],
                'y': [0, 1, 1, 0],
            },
        },
    ],
    'MarkovChain': [
        {
            'name': 'Simple Greeting',
            'description': 'Generate repeating greetings using a small greeting sample.',
            'args': {'k': 1},
            'training_text': 'hello world hello world hello world',
        },
        {
            'name': 'Lorem Ipsum',
            'description': 'Generate pseudo-Latin text using a Lorem Ipsum excerpt.',
            'args': {'k': 2},
            'training_text': (
                'lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt'
                ' ut labore et dolore magna aliqua'
            ),
        },
    ],
    'HopfieldNetwork': [
        {
            'name': '4-bit Patterns',
            'description': 'Recall binary patterns of length 4.',
            'args': {'n_units': 4},
            'patterns': [[1, -1, 1, -1], [1, 1, -1, -1]],
        },
        {
            'name': '6-bit Patterns',
            'description': 'Recall binary patterns of length 6.',
            'args': {'n_units': 6},
            'patterns': [[1, -1, 1, -1, 1, -1], [-1, 1, -1, 1, -1, 1]],
        },
    ],
}

def list_model_classes():
    """Discover models in the models_nn package."""
    try:
        model_pkg = importlib.import_module('models_nn')
    except ImportError:
        console.print('[red]Could not import models_nn package.[/]')
        return []
    classes = []
    for finder, name, ispkg in pkgutil.iter_modules(model_pkg.__path__):
        module_name = f'models_nn.{name}'
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for attr_name, attr in inspect.getmembers(module, inspect.isclass):
            if attr.__module__ == module_name:
                classes.append((attr_name, attr))
    return classes

def select_model(classes):
    """Prompt user to select a model from the list."""
    table = Table(title='Available Models')
    table.add_column('Index', style='cyan', no_wrap=True)
    table.add_column('Model', style='magenta')
    for idx, (name, _) in enumerate(classes):
        table.add_row(str(idx), name)
    console.print(table)
    choice = Prompt.ask('Select a model by index', choices=[str(i) for i in range(len(classes))])
    return classes[int(choice)][1]

def get_init_params(cls):
    """Inspect constructor and prompt for parameters."""
    sig = inspect.signature(cls.__init__)
    params = {}
    for name, param in sig.parameters.items():
        if name == 'self':
            continue
        ann = param.annotation
        default = param.default
        prompt_text = f'{name}'
        if default is inspect._empty:
            if ann != inspect._empty:
                type_name = getattr(ann, '__name__', str(ann))
                prompt_text += f' (type: {type_name})'
            value = Prompt.ask(prompt_text)
        else:
            prompt_text += f' (default: {default})'
            value = Prompt.ask(prompt_text, default=str(default))
        # Convert to appropriate type
        if default is inspect._empty:
            if ann == int:
                params[name] = int(value)
            elif ann == float:
                params[name] = float(value)
            else:
                params[name] = value
        else:
            if isinstance(default, bool):
                params[name] = value.lower() in ('y', 'yes', 'true', '1')
            elif isinstance(default, int):
                try:
                    params[name] = int(value)
                except ValueError:
                    params[name] = default
            elif isinstance(default, float):
                try:
                    params[name] = float(value)
                except ValueError:
                    params[name] = default
            else:
                params[name] = value
    return params

def run_perceptron(model, preset=None):
    """Run perceptron demo, using preset training data if provided."""
    if preset and 'training_data' in preset:
        console.print(Panel(preset['description'], title=preset['name'], style='green'))
        data = preset['training_data']
        X, y = data.get('X', []), data.get('y', [])
    else:
        console.print(Panel('Perceptron: Learning OR gate example', style='green'))
        X = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y = [0, 1, 1, 1]
    model.train(X, y, epochs=10)
    results = [model.predict(x) for x in X]
    console.print('Predictions:')
    for xi, yi in zip(X, results):
        console.print(f'  {xi} -> {yi}')
    console.print(Panel.fit(model.visualize(), title='Perceptron Structure'))

def run_markov_chain(model, preset=None):
    """Run Markov Chain demo, using preset training text if provided."""
    if preset and 'training_text' in preset:
        console.print(Panel(preset['description'], title=preset['name'], style='green'))
        text = preset['training_text']
    else:
        console.print(Panel('Markov Chain: Text generation', style='green'))
        text = Prompt.ask('Enter training text (leave blank for default)', default='')
        if not text:
            text = 'hello world'
    model.train(text)
    length_str = Prompt.ask('Length of generated text', default='50')
    try:
        length = int(length_str)
    except ValueError:
        length = 50
    seed = Prompt.ask(f'Seed (length {model.k}, optional)', default='') or None
    output = model.generate(length=length, seed=seed)
    console.print(Panel(output, title='Generated Text'))
    console.print(Panel.fit(model.visualize(), title='Markov Chain Structure'))

def run_hopfield(model, preset=None):
    """Run Hopfield Network demo, using preset patterns if provided."""
    if preset and 'patterns' in preset:
        console.print(Panel(preset['description'], title=preset['name'], style='green'))
        patterns = preset['patterns']
        model.train(patterns)
        console.print(Panel.fit(model.visualize(), title='Network Structure'))
        # Demonstrate recall with a corrupted version of the first pattern
        original = patterns[0]
        corrupted = list(original)
        # flip the first bit for corruption
        corrupted[0] = -corrupted[0]
        result = model.recall(corrupted)
        console.print(f'Original pattern: {original}')
        console.print(f'Corrupted pattern: {corrupted}')
        console.print(f'Recalled pattern: {result}')
    else:
        console.print(Panel('Hopfield Network: Associative memory', style='green'))
        console.print(Panel.fit(model.visualize(), title='Network Structure'))
        # Prompt for a pattern to recall
        pattern_str = Prompt.ask(f'Enter comma-separated ±1 pattern of length {model.n_units} (blank to skip)', default='')
        if pattern_str:
            try:
                pattern = [int(x.strip()) for x in pattern_str.split(',')]
                result = model.recall(pattern)
                console.print(f'Recalled pattern: {result}')
            except Exception as e:
                console.print(f'[red]Error recalling pattern: {e}[/]')

def main():
    console.print(Text('Neural Network Model Explorer', style='bold underline'))
    classes = list_model_classes()
    if not classes:
        console.print('[red]No models found.[/]')
        sys.exit(1)
    cls = select_model(classes)
    cls_name = cls.__name__
    # Choose from predefined examples if available
    presets = MODEL_PRESETS.get(cls_name)
    if presets:
        # list example presets
        p_table = Table(title=f'{cls_name} Examples')
        p_table.add_column('Index', style='cyan', no_wrap=True)
        p_table.add_column('Example', style='magenta')
        p_table.add_column('Description', style='green')
        for idx, p in enumerate(presets):
            p_table.add_row(str(idx), p['name'], p['description'])
        console.print(p_table)
        choice = Prompt.ask(f'Select an example for {cls_name}', choices=[str(i) for i in range(len(presets))])
        preset = presets[int(choice)]
        console.print(Panel(preset['description'], title=preset['name'], style='green'))
        model = cls(**preset.get('args', {}))
        # run the demo with selected preset
        if cls_name == 'Perceptron':
            run_perceptron(model, preset)
        elif cls_name == 'MarkovChain':
            run_markov_chain(model, preset)
        elif cls_name == 'HopfieldNetwork':
            run_hopfield(model, preset)
        else:
            console.print(Panel.fit(model.visualize(), title=f'{cls_name} Visualization'))
        return
    # fallback to manual parameter entry for other models
    params = get_init_params(cls)
    model = cls(**params)
    if cls_name == 'Perceptron':
        run_perceptron(model)
    elif cls_name == 'MarkovChain':
        run_markov_chain(model)
    elif cls_name == 'HopfieldNetwork':
        run_hopfield(model)
    else:
        console.print(f'No interactive demo available for {cls_name}.')
        console.print(Panel.fit(model.visualize(), title=f'{cls_name} Visualization'))

if __name__ == '__main__':
    main()
