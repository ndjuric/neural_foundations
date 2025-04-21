#!/usr/bin/env python
import requests
import sys
from perceptron import Perceptron
from hopfield_network import HopfieldNetwork
from markov_chain import MarkovChain

class ExampleRunner:
    """
    A class to encapsulate and run examples for Perceptron, Hopfield Network, and Markov Chain.
    """

    def run_perceptron_example(self):
        """
        Demonstrates the Perceptron learning an OR gate.
        """
        try:
            # OR gate
            X = [[0, 0], [0, 1], [1, 0], [1, 1]]
            y = [0, 1, 1, 1]
            p = Perceptron(n_inputs=2, learning_rate=0.1)
            p.train(X, y, epochs=10)
            print("Perceptron OR gate predictions:")
            for xi in X:
                print(f"  {xi} -> {p.predict(xi)}")
            print("\nPerceptron structure (Mermaid):")
            print(p.visualize())
        except Exception as e:
            print(f"An error occurred in the Perceptron example: {e}")

    def run_hopfield_example(self):
        """
        Demonstrates the Hopfield Network recalling a corrupted pattern.
        """
        try:
            patterns = [[1, -1, 1, -1], [1, 1, -1, -1]]
            hn = HopfieldNetwork(n_units=4)
            hn.train(patterns)
            test_state = [1, -1, -1, -1]
            recalled = hn.recall(test_state)
            print("Hopfield recall of corrupted pattern [1, -1, -1, -1]:")
            print(f"  {test_state} -> {recalled}")
            print("\nHopfield structure (Mermaid):")
            print(hn.visualize())
        except Exception as e:
            print(f"An error occurred in the Hopfield Network example: {e}")

    def run_markov_chain_example(self):
        """
        Demonstrates the Markov Chain generating text based on training data.
        """
        try:
            print("Do you want to download an example training text from Project Gutenberg? (y/n)")
            choice = input("> ").strip().lower()

            if choice == 'y':
                # Download example text
                url = "https://www.gutenberg.org/files/1342/1342-0.txt"
                print(f"Downloading text from {url}...")
                response = requests.get(url, timeout=10)
                response.raise_for_status()  # Raise an error for HTTP issues
                text = response.text[:10000]  # Use only the first 10,000 characters for simplicity
                print("\nDownloaded text (first 500 characters):")
                print(text[:500])
            else:
                # Prompt user to input their own text
                print("Please input your own training text (or paste a paragraph):")
                text = input("> ").strip()
                if not text:
                    print("No text provided. Exiting Markov Chain example.")
                    return

            # Train and generate text using Markov Chain
            mc = MarkovChain(k=1)
            mc.train(text)
            generated = mc.generate(length=50)
            print("\nMarkov chain generated text:")
            print(generated)
            print("\nMarkov chain structure (Mermaid):")
            print(mc.visualize())
        except requests.exceptions.RequestException as e:
            print(f"Failed to download the example text: {e}")
        except Exception as e:
            print(f"An error occurred in the Markov Chain example: {e}")

def main():
    """
    Main function to run all examples using the ExampleRunner class.
    """
    try:
        runner = ExampleRunner()
        print("--- Perceptron Example ---")
        runner.run_perceptron_example()
        print("\n--- Hopfield Network Example ---")
        runner.run_hopfield_example()
        print("\n--- Markov Chain Example ---")
        runner.run_markov_chain_example()
    except KeyboardInterrupt:
        print("\nProgram interrupted. Exiting cleanly.")
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()