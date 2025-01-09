# Updated function to format keypoints for one person
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from PIL import Image 
import ast



def formatkps(row):
    keypoints = []
    for part in ['head', 'neck', 'left_ear', 'right_ear', 'left_shoulder', 'left_elbow', 
                 'left_hand', 'right_shoulder', 'right_elbow', 'right_hand']:
        try:
            x, y = ast.literal_eval(row[part])
        except:
            x, y = -1, -1
        keypoints.append(x)
        keypoints.append(y)
    return keypoints


class SinglePersonDataset(Dataset):
    def __init__(self, data, images_folder, transform=None):
        self.data = data
        self.images_folder = images_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load and preprocess image
        # image_path = os.path.join(self.images_folder, f"frame_{int(row['frame']):04d}.jpg")
        try:
            frame_number = int(row['frame'])
            image_path = os.path.join(self.images_folder, f"frame_{frame_number:04d}.jpg")
        
        except ValueError:
            print(f"Invalid frame value: {row['frame']}. Skipping.")
            return None  # Skip this sample
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}. Skipping.")
            return None

        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}. Skipping.")
            return None

        # Load keypoints
        keypoints = torch.tensor(formatkps(row), dtype=torch.float32)

        # Load label
        label = torch.tensor(row.get('hand_raised', 0), dtype=torch.float32).unsqueeze(0)

        return image, keypoints, label


# Updated hybrid model for one person
class SinglePersonModel(nn.Module):
    def __init__(self, image_feature_dim=1280, keypoints_dim=20, num_classes=1):
        super(SinglePersonModel, self).__init__()

        # Pretrained EfficientNet
        self.image_model = models.efficientnet_b0(pretrained=True)
        self.image_model.classifier = nn.Identity()

        # Keypoint processing
        self.keypoints_fc = nn.Sequential(
            nn.Linear(keypoints_dim, 64),
            nn.ReLU(),
        )

        # Combined features
        self.fc = nn.Sequential(
            nn.Linear(image_feature_dim + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, image, keypoints):
        image_features = self.image_model(image)
        keypoints_features = self.keypoints_fc(keypoints)
        combined = torch.cat((image_features, keypoints_features), dim=1)
        return self.fc(combined)

# Update training and validation to handle single-person labels
# ... No changes needed in the training loop as the logic adapts to single-label output ...
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # for images, keypoints, labels in dataloader:
    #     if images is None:
    #         continue  # Skip invalid batches
    for batch in dataloader:
        if batch is None:
            continue
    
        images, keypoints, labels = batch 
        images, keypoints, labels = images.to(device), keypoints.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, keypoints)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total += labels.size(0)
        correct += ((outputs > 0.5) == labels).sum().item()
    
    if total == 0:  # Handle case where no valid samples were processed
        return float('inf'), 0.0

    return total_loss / total, correct / total

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        # for images, keypoints, labels in dataloader:
        #     if images is None:
        #         continue  # Skip invalid batches
        for batch in dataloader:
            if batch is None:
                continue
            
            images, keypoints, labels = batch
            images, keypoints, labels = images.to(device), keypoints.to(device), labels.to(device)

            outputs = model(images, keypoints)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            total += labels.size(0)
            correct += ((outputs > 0.5) == labels).sum().item()

    if total == 0:  # Handle case where no valid samples were processed
        return float('inf'), 0.0

    return total_loss / total, correct / total

def custom_collate_fn(batch):
    # Filter out None samples
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None  # Return None if the entire batch is invalid

    images, keypoints, labels = zip(*batch)
    return (
        torch.stack(images),
        torch.stack(keypoints),
        torch.stack(labels)
    )

if __name__ == "__main__":
    # Define paths and data
    images_folder = "C:/OsamaEjaz/Qiyas_Gaze_Estimation/Wajahat_Yolo_keypoint/frames_output/cropped_output/frmaes/"
    csv_file = "C:/OsamaEjaz/Qiyas_Gaze_Estimation/Wajahat_Yolo_keypoint/frames_output/cropped_output/ssdata.csv"
    data = pd.read_csv(csv_file)

    # Train-validation split
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    # Define transforms
    transform = transforms.Compose([
        # transforms.Resize((720, 1280)),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create datasets and dataloaders
    train_dataset = SinglePersonDataset(train_data, images_folder, transform=transform)
    val_dataset = SinglePersonDataset(val_data, images_folder, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

    # Initialize model, criterion, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SinglePersonModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    # ... No changes needed ...

    num_epochs = 200
    early_stopping_patience = 25
    best_val_loss = float('inf')
    stopping_counter = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "ss_hybrid_model.pth")
            stopping_counter = 0
        else:
            stopping_counter += 1

        if stopping_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    print("Training complete. Best model saved as 'ss_hybrid_model.pth'.")
