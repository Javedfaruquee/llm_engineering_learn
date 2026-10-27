"""Inference-only copy of the Week 6 deep neural network pricer.

The network was trained in Week 6 (see week6/pricer/deep_neural_network.py); its weights
are saved in deep_neural_network.pth. This module rebuilds the same architecture, loads
those weights and turns product text into a price. Used by NeuralNetworkAgent.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import HashingVectorizer
import logging


class ResidualBlock(nn.Module):
    """
    Two linear layers with LayerNorm, ReLU and dropout, whose output is added back to
    the block's input (a skip connection). The skip lets gradients flow straight through,
    which is what makes very deep networks trainable.
    """

    def __init__(self, hidden_size, dropout_prob):
        """hidden_size is the width in and out; dropout_prob the share of units dropped in training"""
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        """Return relu(block(x) + x)"""
        residual = x
        out = self.block(x)
        out += residual  # Skip connection
        return self.relu(out)


class DeepNeuralNetwork(nn.Module):
    """
    Input layer, then num_layers - 2 residual blocks of width hidden_size, then a single
    output. With the defaults (10 layers, width 4096) it has about 290 million parameters.
    Must match the Week 6 architecture exactly, or the saved weights won't load.
    """

    def __init__(self, input_size, num_layers=10, hidden_size=4096, dropout_prob=0.2):
        """input_size is the length of the bag-of-words vectors"""
        super(DeepNeuralNetwork, self).__init__()

        # First layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(num_layers - 2):
            self.residual_blocks.append(ResidualBlock(hidden_size, dropout_prob))

        # Output layer
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """Map a batch of vectors (batch, input_size) to normalised log prices (batch, 1)"""
        x = self.input_layer(x)

        for block in self.residual_blocks:
            x = block(x)

        return self.output_layer(x)


# Mean and standard deviation of log(price + 1) over the Week 6 training set. The network
# predicts (log(price + 1) - Y_MEAN) / Y_STD, so these undo that to get back to dollars.
Y_STD = 1.0328539609909058
Y_MEAN = 4.434937953948975


class DeepNeuralNetworkInference:
    """Loads the trained network and prices product text with it.

    Usage: setup() once, load(path) the weights, then inference(text) per product.
    """

    def __init__(self):
        """Fix the random seeds; the vectorizer, model and device are created in setup()."""
        self.vectorizer = None
        self.model = None
        self.device = None

        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    def setup(self):
        """Create the text vectorizer and an untrained network on the best device available.

        The vectorizer must use the same settings as in training: 5000 hashed word features,
        English stop words removed, 1/0 for present/absent.
        """
        self.vectorizer = HashingVectorizer(n_features=5000, stop_words="english", binary=True)
        self.model = DeepNeuralNetwork(5000)
        # Prefer an NVIDIA GPU, then Apple Silicon, then the CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        logging.info(f"Neural Network is using {self.device}")

        self.model.to(self.device)

    def load(self, path):
        """Load the trained weights from path (a .pth file saved in Week 6)."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)

    def inference(self, text):
        """Return the predicted price of the product described by text, in dollars (never negative)."""
        self.model.eval()                   # switch off dropout
        with torch.no_grad():
            vector = self.vectorizer.transform([text])
            vector = torch.FloatTensor(vector.toarray()).to(self.device)
            pred = self.model(vector)[0]
            # Undo the normalisation and the log: price = exp(pred * std + mean) - 1
            result = torch.exp(pred * Y_STD + Y_MEAN) - 1
            result = result.item()
        return max(0, result)
