from load_data import load_data
from split_data import split_data
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np

# load and preprocess data
preprocessed_data = load_data()

# split data
X = preprocessed_data['TEXT']
y = preprocessed_data['AGE']

# subset for faster training
subset_size = 100
# for whole training dataset: X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X[:subset_size], y[:subset_size]) # for subset

# print X_train, X_val, X_test, y_train, y_val, and y_test for modeling tasks
print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

# initialize clinical BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# tokenize and encode text
X_train_encoded = tokenizer(list(X_train), padding=True, truncation=True, return_tensors="pt", max_length=512)
X_val_encoded = tokenizer(list(X_val), padding=True, truncation=True, return_tensors="pt", max_length=512)

# output = model(input_ids=X_train_encoded['input_ids'][0].unsqueeze(0), attention_mask=X_train_encoded['attention_mask'][0].unsqueeze(0))

class AgePredictionModel(nn.Module):
    def __init__(self, bert_model):
        super(AgePredictionModel, self).__init__()
        self.bert = bert_model
        self.fc = nn.Linear(bert_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = outputs.pooler_output
        age_prediction = self.fc(pooler_output)
        return age_prediction.view(-1)

# instantiate the model
age_model = AgePredictionModel(model)

# define loss and optimizer
criterion = nn.MSELoss()  # mean squared error for regression
optimizer = optim.Adam(age_model.parameters(), lr=0.0001)

# convert data to PyTorch tensors
X_train_ids = X_train_encoded['input_ids'].clone()
X_train_mask = X_train_encoded['attention_mask'].clone()
y_train = np.array(y_train, dtype=np.float32)
y_train = torch.tensor(y_train, dtype=torch.float).view(-1)

X_val_ids = X_val_encoded['input_ids'].clone()
X_val_mask = X_val_encoded['attention_mask'].clone()
y_val = torch.tensor(y_val.values, dtype=torch.float).view(-1)

# create dataloader objects
train_data = TensorDataset(X_train_ids, X_train_mask, y_train)
train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)

val_data = TensorDataset(X_val_ids, X_val_mask, y_val)
val_dataloader = DataLoader(val_data, batch_size=16)

# training loop
num_epochs = 5  # adjust as needed
for epoch in range(num_epochs):
    age_model.train()
    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        output = age_model(input_ids, attention_mask)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / input_ids.shape[0] # keep?
    # avg_loss = total_loss / len(train_dataloader)

    # validation
    age_model.eval()
    val_loss = 0
    predictions = []
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_mask, labels = batch
            output = age_model(input_ids, attention_mask)
            val_loss += criterion(output, labels).item()
            predictions.extend(output.tolist())
    avg_val_loss = val_loss / input_ids.shape[0] # keep?
    # avg_val_loss = val_loss / len(val_dataloader)

    # calculate validation metrics
    val_rmse = sqrt(avg_val_loss)  # root mean squared error

    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_loss:.4f}, Val RMSE: {val_rmse:.4f}")

# evaluation on the test set
X_test_encoded = tokenizer(list(X_test), padding=True, truncation=True, return_tensors="pt", max_length=512)
X_test_ids = X_test_encoded['input_ids'].clone()
X_test_mask = X_test_encoded['attention_mask'].clone()
y_test = torch.tensor(y_test, dtype=torch.float)

test_data = TensorDataset(X_test_ids, X_test_mask, y_test)
test_dataloader = DataLoader(test_data, batch_size=16)

age_model.eval()
test_loss = 0
predictions = []
with torch.no_grad():
    for batch in test_dataloader:
        input_ids, attention_mask, labels = batch
        output = age_model(input_ids, attention_mask)
        test_loss += criterion(output, labels).item()
        predictions.extend(output.tolist())
test_rmse = sqrt(test_loss)

print(f"Test RMSE: {test_rmse:.4f}")

# save the trained model weights for future use
torch.save(age_model.state_dict(), 'age_prediction_model.pth')