from load_data import load_data
from split_data import split_data
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from math import sqrt
import numpy as np

# load and preprocess data
preprocessed_data = load_data()

# split data
#ogX = preprocessed_data['TEXT']
X = preprocessed_data[['SUBJECT_ID', 'TEXT']]
y = preprocessed_data['AGE']

# subset for faster training
subset_size = 100
# for whole training dataset: X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
#ogX_train, ogX_val, ogX_test, ogy_train, ogy_val, ogy_test = split_data(ogX[:subset_size], y[:subset_size]) # for subset
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X[:subset_size], y[:subset_size]) # for subset

# print X_train, X_val, X_test, y_train, y_val, and y_test for modeling tasks
print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

# initialize clinical BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# somehow need to pass through Subject Ids without encoding them
# tokenize and encode text
X_train_encoded = tokenizer(list(X_train['TEXT']), padding=True, truncation=True, return_tensors="pt", max_length=512)
X_val_encoded = tokenizer(list(X_val['TEXT']), padding=True, truncation=True, return_tensors="pt", max_length=512)

# output = model(input_ids=X_train_encoded['input_ids'][0].unsqueeze(0), attention_mask=X_train_encoded['attention_mask'][0].unsqueeze(0))

class AgePredictionModel(nn.Module):
    def __init__(self, bert_model):
        super(AgePredictionModel, self).__init__()
        self.bert = bert_model
        for param in self.bert.parameters():
            param.requires_grad = False  # Freezing gradients for BERT
        # decrease complexity -> dropout
        self.fc1 = nn.Linear(bert_model.config.hidden_size, 512) # 1 with no added layers
        # increase complexity
        self.relu = nn.ReLU() # add activation function between
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = outputs.pooler_output
        # age_prediction = self.fc(pooler_output)
        # return age_prediction.view(-1)

        # pass through layers
        x = self.fc1(pooler_output)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        age_prediction = x.squeeze(-1)
        return age_prediction

# instantiate the model
age_model = AgePredictionModel(model)

parameter_count = sum(p.numel() for p in age_model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {parameter_count}")

# define loss and optimizer
criterion = nn.MSELoss()  # mean squared error for regression
optimizer = optim.Adam(age_model.parameters(), lr=0.01)

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
num_epochs = 1
for epoch in range(num_epochs):
    age_model.train()
    total_loss = 0
    total_examples = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        output = age_model(input_ids, attention_mask)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_examples += input_ids.shape[0]
    train_rmse = sqrt(total_loss / total_examples)

    # validation
    age_model.eval()
    val_loss = 0
    val_examples = 0
    predictions = []
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_mask, labels = batch
            output = age_model(input_ids, attention_mask)
            val_loss += criterion(output, labels).item()
            val_examples += input_ids.shape[0]
            predictions.extend(output.tolist())
    val_rmse = sqrt(val_loss / val_examples)

    print(f"Epoch {epoch+1}/{num_epochs}: Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}")

# evaluation on the test set
X_test_encoded = tokenizer(list(X_test), padding=True, truncation=True, return_tensors="pt", max_length=512)
X_test_ids = X_test_encoded['input_ids'].clone()
X_test_mask = X_test_encoded['attention_mask'].clone()
y_test = torch.tensor(y_test, dtype=torch.float)

test_data = TensorDataset(X_test_ids, X_test_mask, y_test)
test_dataloader = DataLoader(test_data, batch_size=16)

age_model.eval()
test_loss = 0
test_examples = 0
predictions = []
true_labels = []
with torch.no_grad():
    for batch in test_dataloader:
        input_ids, attention_mask, labels = batch
        output = age_model(input_ids, attention_mask)
        test_loss += criterion(output, labels).item()
        test_examples += input_ids.shape[0]
        predictions.extend(output.tolist())
        true_labels.extend(labels.tolist())
test_rmse = sqrt(test_loss / test_examples)

print(f"Test RMSE: {test_rmse:.4f}")

print("Predictions\tTrue Ages")
for pred, true_age in zip(predictions, true_labels):
    print(f"{pred:.4f}\t{true_age:.4f}")

# save the trained model weights for future use
#torch.save(age_model.state_dict(), 'age_prediction_model.pth')
