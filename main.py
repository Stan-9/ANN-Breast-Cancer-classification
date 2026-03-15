import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # 1. Load the dataset
    print("Loading Breast Cancer dataset...")
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names

    # 2. Preprocess the data
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

    # 3. Define the ANN Architecture
    class SimpleANN(nn.Module):
        def __init__(self, input_dim):
            super(SimpleANN, self).__init__()
            self.layer1 = nn.Linear(input_dim, 16)
            self.relu1 = nn.ReLU()
            self.layer2 = nn.Linear(16, 8)
            self.relu2 = nn.ReLU()
            self.output = nn.Linear(8, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.relu1(self.layer1(x))
            x = self.relu2(self.layer2(x))
            x = self.sigmoid(self.output(x))
            return x

    input_dim = X.shape[1]
    model = SimpleANN(input_dim)
    print(f"Model Architecture:\n{model}")

    # 4. Define Loss and Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 5. Train the model
    num_epochs = 100
    train_losses = []
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # 6. Evaluation
    print("Evaluating model...")
    model.eval()
    with torch.no_grad():
        y_pred_prob = model(X_test_tensor)
        y_pred = (y_pred_prob > 0.5).float()
    
    # Convert back to numpy for metrics
    y_test_np = y_test_tensor.numpy()
    y_pred_np = y_pred.numpy()

    print("\nClassification Report:")
    print(classification_report(y_test_np, y_pred_np, target_names=data.target_names))

    # 7. Visualizations
    # Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    print("Loss curve saved as 'loss_curve.png'")

    # Confusion Matrix
    cm = confusion_matrix(y_test_np, y_pred_np)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")

if __name__ == "__main__":
    main()
