<h1>QuantAnalysisAi</h1>


<h2>Description</h2>
This script shows the Training of LSTM models creating an AI that analyses the market on hoistorical data, trains on it on the very granular activity of the market such that it can predict the future movement of the market, only this time using volatilioty indices which are a simulation of the market movement in the real world.
<br />


<h2>Languages and Utilities Used</h2>

- <b>Python</b>

<h2>Environments Used </h2>

- <b>Google colab</b> 

<h2>Program walk-through:</h2>

<p align="center">
Login, put token and import libraries output: <br/>
<img src="https://i.imgur.com/87AQVem.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />
<br />
When done and no error found will print setup complete:  <br/>
<img src="https://i.imgur.com/FsVLeho.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />
<br />
Load historical data: <br/>
<img src="https://i.imgur.com/4PnErca.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />
<br />
Initialize the models for training:  <br/>
<img src="https://i.imgur.com/dakcVqw.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />
<br />
Train the models on the  historical data (may take some time):  <br/>
<img src="https://i.imgur.com/ELe7g9J.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />
<br />
Finish the Quanttraining and save the models to google drive and rerun if accuracy is not up to your liking:  <br/>
<img src="https://i.imgur.com/xVin798.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />

```THE CODE
# Cell 1: Core Setup - RUN FIRST !!

# Install the required version of websockets first
!pip install "websockets>=13.0,<15.0dev"

# Install other necessary packages
!pip install pandas
!pip install torch_geometric
!pip install python_deriv_api
!pip install nest_asyncio
!pip install rx
!pip install sklearn
!pip install torch_optimizer
!pip install tensorflow

# Import necessary libraries
import nest_asyncio
import asyncio
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Attention, Bidirectional
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score
from deriv_api import DerivAPI

# Replace 'YOUR_ACTUAL_TOKEN' with your real Deriv API token
api = DerivAPI(app_id=xxxx)

async def authorize_api():
    try:
        response = await api.authorize('xxxxx')  # token
        print('✅ Authorization complete.')
        print('Response details:', json.dumps(response, indent=4))  # Print the full response in a readable format
    except Exception as e:
        print('❌ Authorization failed.')
        print(f'Error message: {e}')
        if hasattr(e, 'response'):
            print(f"API Response: {e.response.text}")

# Integrate with the existing event loop using nest_asyncio
nest_asyncio.apply()

# Run the authorization function
asyncio.run(authorize_api())

print('✅ Setup complete. Ready to proceed with the next steps.')



#cell 2 market data engine... loading data

class MarketData:
    def __init__(self):
        self.training_years = 3
        self.window_size = 256
        self.batch_size = 64
        self.symbols = ['R_100']  # Volatility symbols identified
        self.historical_data = {}

    async def load_historical(self):
        print("Loading historical data...")

        # Calculate the start time for 2 years ago
        # Get current timestamp using pd.Timestamp.now()
        end_time = pd.Timestamp.now().timestamp()
        start_time = end_time - (2 * 365 * 24 * 60 * 60)  # 2 years ago in seconds

        for symbol in self.symbols:
            try:
                response = await api.ticks_history({
                    "ticks_history": symbol,
                    "adjust_start_time": 1,
                    "start": int(start_time),  # Start time 2 years ago
                    "end": "latest",           # End time now
                    "style": "ticks"
                })

                # print(f"Response for {symbol}: {response}")  # Remove detailed response printing

                if 'history' in response and 'prices' in response['history'] and 'times' in response['history']:
                    prices = response['history']['prices']
                    times = response['history']['times']
                    self.historical_data[symbol] = pd.DataFrame({
                        'price': prices,
                        'epoch': times
                    })
                else:
                    print(f"Unexpected response format for {symbol}: {response}")

            except Exception as e:
                print(f"Error loading data for {symbol}: {e}")

        print("Data loaded.")  # Print "Data loaded"
        # print("Function executed") # Remove extra print statement
        # return self.historical_data # No need to return data here

# Create an instance of MarketData outside the main function
data_engine = MarketData()

async def main():
    global data_engine
    await data_engine.load_historical()  # Just load the data

asyncio.run(main())

#Cell 3 Initialization of the models

import tensorflow as tf

class TemporalNet(tf.keras.Model):
    """LSTM with temporal attention (trained on price history)"""
    def __init__(self):
        super().__init__(name="TemporalNet")
        self.conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')  # Convolutional layer
        self.lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))  # Bidirectional LSTM
        self.attention = tf.keras.layers.Attention()  # Attention layer
        self.lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))  # Second Bidirectional LSTM
        self.dropout = tf.keras.layers.Dropout(0.5)  # Dropout for regularization
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer

    def call(self, inputs):
        x = self.conv1(inputs)  # Apply convolution to input sequence
        x = self.lstm1(x)  # First Bidirectional LSTM
        x = self.attention([x, x])  # Apply attention
        x = self.lstm2(x)  # Second Bidirectional LSTM
        x = self.dropout(x)  # Apply dropout
        return self.dense(x)  # Output prediction

# Initialize the pre-trained model (weights can be loaded here if available)
lstm_model = TemporalNet()
print("🧠 Model initialized - ready for real-time analysis")

#Cell4 training

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader

from google.colab import drive
# Try mounting with force_remount=True to resolve potential issues
drive.mount('/content/drive', force_remount=True)

class DataEngine:
    def __init__(self, historical_data, window_size, batch_size):
        self.historical_data = historical_data
        self.window_size = window_size
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.symbols = list(historical_data.keys())

    def normalize_features(self, df):
        return pd.DataFrame(self.scaler.fit_transform(df), columns=df.columns)

    def create_dataset(self, symbol):
        df = self.historical_data[symbol]
        df_normalized = self.normalize_features(df[['price']])
        sequences = []
        labels = []

        for i in range(len(df) - self.window_size):
            seq = df_normalized.iloc[i:i + self.window_size].values
            label = 1 if df['price'].iloc[i + self.window_size] > df['price'].iloc[i + self.window_size - 1] else 0
            sequences.append(seq)
            labels.append(label)

        return np.array(sequences), np.array(labels)

class TimeSeriesDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, label

class AttentionLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, output_size=1, dropout_rate=0.3):
        super(AttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)

        # Calculate context vector using attention weights
        context_vector = torch.bmm(attention_weights.permute(0, 2, 1), lstm_out)

        # Squeeze context_vector along dim=1
        context_vector = context_vector.squeeze(1)

        # Pass the context vector through the fully connected layer
        out = self.fc(self.dropout(context_vector))
        return out

class ModelTrainer:
    def __init__(self, model, data_engine):
        self.model = model
        self.data_engine = data_engine
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCEWithLogitsLoss()

    def train(self, epochs=25, accumulation_steps=2):
        self.model.train()

        # Combine datasets for all symbols
        all_features = []
        all_labels = []
        for symbol in self.data_engine.symbols:
            features, labels = self.data_engine.create_dataset(symbol)
            all_features.extend(features)
            all_labels.extend(labels)

        dataset = TimeSeriesDataset(np.array(all_features), np.array(all_labels))
        dataloader = DataLoader(dataset, batch_size=self.data_engine.batch_size, shuffle=True)

        for epoch in range(epochs):
            print(f"Starting epoch {epoch + 1}...")  # Print epoch update
            total_loss = 0
            for i, (features, labels) in enumerate(dataloader):
                preds = self.model(features)
                loss = self.criterion(preds, labels.unsqueeze(1))
                loss = loss / accumulation_steps
                loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            predicted_labels = (torch.sigmoid(preds) > 0.5).float()
            accuracy = accuracy_score(labels.cpu().detach().numpy(), predicted_labels.cpu().detach().numpy())
            f1 = f1_score(labels.cpu().detach().numpy(), predicted_labels.cpu().detach().numpy())  # Calculate F1-score

            print(f"Epoch {epoch + 1}: Loss {avg_loss:.4f}, Accuracy {accuracy:.4f}")  # Print epoch results

        # Print final evaluation after all epochs
        print(f"Finished training. Final Evaluation: Loss {avg_loss:.4f}, Accuracy {accuracy:.4f}, F1 Score {f1:.4f}")

# Training and Evaluation
window_size = 256
batch_size = 32
data_engine = DataEngine(data_engine.historical_data, window_size, batch_size)

model = AttentionLSTM(input_size=1, hidden_size=128, output_size=1)
trainer = ModelTrainer(model, data_engine)

trainer.train(epochs=25, accumulation_steps=2)  # Train the model for all symbols combined

# Save the model
model_save_path = '/content/drive/My Drive/wednesday.pth'  # Change the path as needed
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}') ```

<!--
 ```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@
```
--!>
