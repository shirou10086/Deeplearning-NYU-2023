import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AdamW
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# Assuming 'load_and_preprocess_data' function is defined in 'data_preparation.py'
from data_preparation import load_and_preprocess_data

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def train_and_evaluate(dataset_path, model_path, num_labels):
    # Load and preprocess data
    train_encodings, train_labels, test_encodings, test_labels = load_and_preprocess_data(dataset_path)

    # Create datasets
    train_dataset = CustomDataset(train_encodings, train_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Load pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", num_labels=num_labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Loss function
    loss_function = CrossEntropyLoss()

    # Track loss values for plotting
    # Track loss and accuracy values for plotting
    epoch_loss_values = []
    epoch_accuracy_values = []

    # Train model
    num_epochs = 10  # Set the number of epochs
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for batch in tqdm(train_loader):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(**inputs)
            loss = loss_function(outputs.logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        epoch_loss_values.append(avg_loss)

        epoch_accuracy = correct_predictions / total_predictions
        epoch_accuracy_values.append(epoch_accuracy)

        print(f"Epoch {epoch+1} - Loss: {avg_loss}, Accuracy: {epoch_accuracy}")

    # Plot training loss and accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_loss_values, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epoch_accuracy_values, label='Training Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Evaluate model
    model.eval()
    predictions, true_labels = [], []
    for batch in test_loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predictions.extend(logits.argmax(dim=-1).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    print(f"Test Accuracy: {accuracy}")

    # Save model
    model.save_pretrained(model_path)

# Ensure that 'data_preparation.py' and necessary libraries are correctly set up.
