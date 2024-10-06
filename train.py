
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchmetrics import FBetaScore

import numpy as np

import random

import h5py


class AuthData(Dataset):
    def __init__(self):
        signatures = []

        with h5py.File('data.h5', 'r') as file:
            for key in file:
                signatures.append(file[key][:])

        positiveSignatures = []
        positiveKeys = []
        positiveLabels = []
        for signature in signatures:
            keyNum = random.randint(0, len(signature) - 1)
            slicedSignature = np.concatenate([signature[:keyNum], signature[keyNum + 1:]])
            positiveSignatures.append(slicedSignature)

            positiveKeys.append(signature[keyNum])

            positiveLabels.append(1)

        negativeSignatures = []
        negativeKeys = []
        negativeLabels = []
        for i in range(len(signatures)):
            keyNum = random.randint(0, len(signatures[i]) - 1)
            slicedSignature = np.concatenate([signatures[i][:keyNum], signatures[i][keyNum + 1:]])
            negativeSignatures.append(slicedSignature)

            choices = [k for k in range(len(signatures)) if k != i]
            wrongNum = random.choice(choices)
            wrongKeyNum = random.randint(0, len(signatures[wrongNum]) - 1)
            negativeKeys.append(signatures[wrongNum][wrongKeyNum])

            negativeLabels.append(0)

        totalSignatures = positiveSignatures + negativeSignatures
        signatureTensors = [torch.tensor(signature, dtype=torch.float32) for signature in totalSignatures]

        totalKeys = positiveKeys + negativeKeys
        keyTensors = [torch.tensor(key, dtype=torch.float32) for key in totalKeys]

        totalLabels = positiveLabels + negativeLabels
        labelTensors = [torch.tensor(label, dtype=torch.float32) for label in totalLabels]

        self.signatures = pad_sequence(signatureTensors, batch_first=True)
        self.keys = keyTensors
        self.labels = labelTensors

    def __len__(self):
        return len(self.signatures)

    def __getitem__(self, idx):
        return self.signatures[idx], self.keys[idx], self.labels[idx]


class ConfidenceModel(nn.Module):
    def __init__(self, inputDim=1024, hiddenDim=512, keyDim=1024, numHeads=4):
        super().__init__()

        self.embedding = nn.Linear(inputDim, hiddenDim)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hiddenDim, numHeads), num_layers=4
        )

        self.fc = nn.Linear(hiddenDim + keyDim, 1)

    def forward(self, inputVectors, keyVector):
        embedded = self.embedding(inputVectors)
        transformed = self.transformer(embedded)

        aggregated = transformed.mean(dim=1)

        combined = torch.cat((aggregated, keyVector), dim=-1)

        confidence = torch.sigmoid(self.fc(combined))
        return confidence


if __name__ == '__main__':
    learning_rate = 0.001
    num_epochs = 100

    dataset = AuthData()

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = ConfidenceModel()  # Adjust dimensions as needed
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    fBeta = FBetaScore(task="binary", beta=0.5)

    for epoch in range(num_epochs):
        currentBestAccuracy = 0

        for batch_inputs, batch_keys, batch_labels in dataloader:
            optimizer.zero_grad()  # Clear gradients

            # Forward pass
            outputs = model(batch_inputs, batch_keys)

            # Compute loss
            loss = criterion(outputs.squeeze(), batch_labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            stepAccuracy = fBeta(outputs.squeeze(), batch_labels)
            currentBestAccuracy = max(currentBestAccuracy, stepAccuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {currentBestAccuracy:.4f}')
