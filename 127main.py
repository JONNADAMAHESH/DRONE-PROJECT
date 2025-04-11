import os
import cv2
import numpy as np
import pandas as pd
import mysql.connector
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import datetime
from yolov5 import train, val

# Set image dimensions and training parameters
IMG_SIZE = 256
IMG_CHANNELS = 3
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 10

# Define categories for the orthophoto feature extraction
categories = ['FARMLAND', 'OPENPLOT-HOUSE', 'ROAD', 'building-road-houses', 'concrete-houses']

# MySQL and MongoDB configurations
mysql_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': 'Mahesh2005',
    'database': 'dron_db'
}

mongo_config = {
    'host': 'localhost',
    'port': 27017,
    'database': 'dron_db'
}

# Check device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Connect to MySQL
mysql_conn = mysql.connector.connect(**mysql_config)
mysql_cursor = mysql_conn.cursor()

# Connect to MongoDB
mongo_client = MongoClient(mongo_config['host'], mongo_config['port'])
mongo_db = mongo_client[mongo_config['database']]
mongo_collection = mongo_db['training_logs']

# Custom Dataset for loading images and masks
class OrthophotoDataset(Dataset):
    def __init__(self, data_dir, categories, img_size, transform=None):
        self.data = []
        self.masks = []
        self.transform = transform
        img_base_path = os.path.join(data_dir, 'images')
        mask_base_path = os.path.join(data_dir, 'masks')

        print(f"Image base path: {img_base_path}")
        print(f"Mask base path: {mask_base_path}")

        for category in categories:
            img_path = os.path.join(img_base_path, category)
            mask_path = os.path.join(mask_base_path, category)

            print(f"Checking image directory: {img_path}")
            print(f"Checking mask directory: {mask_path}")

            if not os.path.isdir(img_path):
                raise FileNotFoundError(f"Image directory does not exist: {img_path}")
            if not os.path.isdir(mask_path):
                raise FileNotFoundError(f"Mask directory does not exist: {mask_path}")

            for img_name in os.listdir(img_path):
                img_file = os.path.join(img_path, img_name)
                mask_file = os.path.join(mask_path, img_name)
                if os.path.isfile(img_file) and os.path.isfile(mask_file):
                    self.data.append(img_file)
                    self.masks.append(mask_file)

        if len(self.data) == 0:
            raise FileNotFoundError("No images or masks found in the specified directories.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image and process it
        img = cv2.imread(self.data[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0  # Normalize
        img = np.transpose(img, (2, 0, 1))  # Convert to CxHxW
        img = torch.FloatTensor(img)

        # Load mask and process it
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        mask = (mask > 127).astype(np.float32)  # Binary mask
        mask = np.expand_dims(mask, axis=0)  # Convert to 1xHxW
        mask = torch.FloatTensor(mask)

        return img, mask

# Load and preprocess data
data_dir = r"C:\Users\jonna\PycharmProjects\pythonProject173\yolov5\runs\train\yolov5_tanmay"
dataset = OrthophotoDataset(data_dir, categories, IMG_SIZE)

# Split dataset into training and validation sets
train_indices, val_indices = train_test_split(
    list(range(len(dataset))), test_size=0.2, random_state=42
)

train_subset = torch.utils.data.Subset(dataset, train_indices)
val_subset = torch.utils.data.Subset(dataset, val_indices)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

# Define the U-Net model for image segmentation
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.c1 = CBR(in_channels, 64)
        self.p1 = nn.MaxPool2d(2)
        self.d1 = nn.Dropout(0.1)

        self.c2 = CBR(64, 128)
        self.p2 = nn.MaxPool2d(2)
        self.d2 = nn.Dropout(0.1)

        self.c3 = CBR(128, 256)
        self.p3 = nn.MaxPool2d(2)
        self.d3 = nn.Dropout(0.2)

        self.c4 = CBR(256, 512)
        self.p4 = nn.MaxPool2d(2)
        self.d4 = nn.Dropout(0.2)

        self.b = CBR(512, 1024)

        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.c6 = CBR(1024, 512)

        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.c7 = CBR(512, 256)

        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.c8 = CBR(256, 128)

        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.c9 = CBR(128, 64)

        self.output = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        c1 = self.c1(x)
        p1 = self.p1(c1)
        p1 = self.d1(p1)

        c2 = self.c2(p1)
        p2 = self.p2(c2)
        p2 = self.d2(p2)

        c3 = self.c3(p2)
        p3 = self.p3(c3)
        p3 = self.d3(p3)

        c4 = self.c4(p3)
        p4 = self.p4(c4)
        p4 = self.d4(p4)

        # Bottleneck
        b = self.b(p4)

        # Decoder
        up6 = self.up6(b)
        up6 = torch.cat([up6, c4], dim=1)
        c6 = self.c6(up6)

        up7 = self.up7(c6)
        up7 = torch.cat([up7, c3], dim=1)
        c7 = self.c7(up7)

        up8 = self.up8(c7)
        up8 = torch.cat([up8, c2], dim=1)
        c8 = self.c8(up8)

        up9 = self.up9(c8)
        up9 = torch.cat([up9, c1], dim=1)
        c9 = self.c9(up9)

        out = self.output(c9)
        out = self.sigmoid(out)
        return out

# Initialize the U-Net model
model = UNet(in_channels=IMG_CHANNELS, out_channels=1).to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Early Stopping and Checkpointing variables
best_val_loss = float('inf')
patience_counter = 0

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Training Loss: {avg_train_loss:.4f} - Validation Loss: {avg_val_loss:.4f}")

    # Checkpointing
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Model saved with validation loss: {avg_val_loss:.4f}")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping!")
            break
