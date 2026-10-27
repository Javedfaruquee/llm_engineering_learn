import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.feature_extraction.text import HashingVectorizer


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


class DeepNeuralNetworkRunner:
    """
    Trains DeepNeuralNetwork to predict prices and uses it as a pricer

    The network predicts a normalised log price, (log(price + 1) - mean) / std, rather
    than dollars: log prices are far less skewed, and normalising keeps targets near 0,
    both of which make training more stable. inference() converts back to dollars.
    Usage: runner = DeepNeuralNetworkRunner(train, val); runner.setup(); runner.train();
    then evaluate(runner.inference, test).
    """

    def __init__(self, train, val):
        """Store the training and validation Items and seed every random generator (42)"""
        self.train_data = train
        self.val_data = val
        self.vectorizer = None
        self.model = None
        self.device = None
        self.loss_function = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataset = None
        self.train_loader = None
        self.y_mean = None
        self.y_std = None

        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    def setup(self):
        """
        Vectorise the data, normalise the prices, build the network and pick a device

        Text becomes 5,000-slot binary bag-of-words vectors (HashingVectorizer). The device
        is CUDA if available (this includes AMD ROCm builds of PyTorch), then Apple MPS,
        then CPU. Loss is L1 (mean absolute error) on the normalised log prices, optimised
        with AdamW and a cosine learning-rate schedule over 10 epochs.
        """
        self.vectorizer = HashingVectorizer(n_features=5000, stop_words="english", binary=True)

        # Inputs and prices as tensors, one row per item
        train_documents = [item.summary for item in self.train_data]
        X_train_np = self.vectorizer.fit_transform(train_documents)
        self.X_train = torch.FloatTensor(X_train_np.toarray())
        y_train_np = np.array([float(item.price) for item in self.train_data])
        self.y_train = torch.FloatTensor(y_train_np).unsqueeze(1)

        val_documents = [item.summary for item in self.val_data]
        X_val_np = self.vectorizer.transform(val_documents)
        self.X_val = torch.FloatTensor(X_val_np.toarray())
        y_val_np = np.array([float(item.price) for item in self.val_data])
        self.y_val = torch.FloatTensor(y_val_np).unsqueeze(1)

        # Targets: log prices, normalised with the training mean and standard deviation
        y_train_log = torch.log(self.y_train + 1)
        y_val_log = torch.log(self.y_val + 1)
        self.y_mean = y_train_log.mean()
        self.y_std = y_train_log.std()
        self.y_train_norm = (y_train_log - self.y_mean) / self.y_std
        self.y_val_norm = (y_val_log - self.y_mean) / self.y_std

        self.model = DeepNeuralNetwork(self.X_train.shape[1])
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Deep Neural Network created with {total_params:,} parameters")

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Using {self.device}")

        self.model.to(self.device)
        self.loss_function = nn.L1Loss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10, eta_min=0)

        self.train_dataset = TensorDataset(self.X_train, self.y_train_norm)
        self.train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)

    def train(self, epochs=5):
        """
        Train for `epochs` passes over the data; after each one print the average training
        loss, the validation loss, the validation error in dollars and the learning rate
        """
        for epoch in range(1, epochs + 1):
            self.model.train()
            train_losses = []

            for batch_X, batch_y in tqdm(self.train_loader):
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.loss_function(outputs, batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                train_losses.append(loss.item())

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(self.X_val.to(self.device))
                val_loss = self.loss_function(val_outputs, self.y_val_norm.to(self.device))

                # Convert back to original scale for meaningful metrics
                val_outputs_orig = torch.exp(val_outputs * self.y_std + self.y_mean) - 1
                mae = torch.abs(val_outputs_orig - self.y_val.to(self.device)).mean()

            avg_train_loss = np.mean(train_losses)
            print(f"Epoch [{epoch}/{epochs}]")
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}")
            print(f"Val mean absolute error: ${mae.item():.2f}")
            print(f"Learning rate: {self.scheduler.get_last_lr()[0]:.6f}")

            self.scheduler.step()

    def save(self, path):
        """Save the trained weights to path"""
        torch.save(self.model.state_dict(), path)

    def load(self, path, device="mps"):
        """
        Load weights saved by save(); call setup() first. `device` is where the saved
        tensors are mapped while loading, before moving them to this runner's device
        """
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.to(self.device)

    def inference(self, item):
        """Predict one item's price in dollars (never below $0); usable as a pricer"""
        self.model.eval()
        with torch.no_grad():
            vector = self.vectorizer.transform([item.summary])
            vector = torch.FloatTensor(vector.toarray()).to(self.device)
            pred = self.model(vector)[0]
            result = torch.exp(pred * self.y_std + self.y_mean) - 1
            result = result.item()
        return max(0, result)
