import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
from load_data import load_data
from split_data import split_data
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from math import sqrt
import numpy as np
import pandas as pd

np.random.seed(1)
torch.manual_seed(1)
torch.use_deterministic_algorithms(True)

# load and preprocess data
preprocessed_data = load_data()

# split data
X = preprocessed_data[['subject_id', 'notes']]
y = preprocessed_data['anchor_age']

# subset for faster training
subset_size = 1000
# for whole training dataset: X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X[:subset_size], y[:subset_size]) # for subset

# print X_train, X_val, X_test, y_train, y_val, and y_test for modeling tasks
print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

X_train['notes'] = X_train['notes'].fillna('')
X_val['notes'] = X_val['notes'].fillna('')
X_test['notes'] = X_test['notes'].fillna('')

# initialize clinical BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# tokenize and encode text
X_train_encoded = tokenizer(list(X_train['notes']), padding=True, truncation=True, return_tensors="pt", max_length=512)
X_val_encoded = tokenizer(list(X_val['notes']), padding=True, truncation=True, return_tensors="pt", max_length=512)

output = model(input_ids=X_train_encoded['input_ids'][0].unsqueeze(0), attention_mask=X_train_encoded['attention_mask'][0].unsqueeze(0))

class AgePredictionModel(nn.Module):
    def __init__(self, bert_model, hidden_layer_size=256):
        super(AgePredictionModel, self).__init__()
        self.bert = bert_model
        for param in self.bert.parameters():
            param.requires_grad = False  # Freezing gradients for BERT
        # decrease complexity -> dropout
        self.fc1 = nn.Linear(bert_model.config.hidden_size, hidden_layer_size) # 1 with no added layers
        # self.fc1 = nn.Linear(64, hidden_layer_size) # 1 with no added layers
        # increase complexity
        self.relu = nn.ReLU() # add activation function between
        # self.relu = nn.Sigmoid() # add activation function between
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_layer_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = outputs.pooler_output
        # pooler_output = outputs.last_hidden_state.max(axis=1)[0]
        # pooler_output = outputs.last_hidden_state.mean(axis=1)
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

device = "cuda:0"

# instantiate the model
age_model = AgePredictionModel(model).to(device)

parameter_count = sum(p.numel() for p in age_model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {parameter_count}")

# define loss and optimizer
criterion = nn.MSELoss()  # mean squared error for regression
optimizer = optim.Adam(age_model.parameters(), lr=0.01)

# convert data to PyTorch tensors
X_train_subject_ids = torch.tensor(X_train['subject_id'].values, dtype=int).to(device)
X_train_ids = X_train_encoded['input_ids'].clone().to(device)
X_train_mask = X_train_encoded['attention_mask'].clone().to(device)
#y_train = np.array(y_train, dtype=np.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float).view(-1).to(device)

X_val_subject_ids = torch.tensor(X_val['subject_id'].values, dtype=int).to(device)
X_val_ids = X_val_encoded['input_ids'].clone().to(device)
X_val_mask = X_val_encoded['attention_mask'].clone().to(device)
y_val = torch.tensor(y_val.values, dtype=torch.float).view(-1).to(device)

# create dataloader objects
train_data = TensorDataset(X_train_subject_ids, X_train_ids, X_train_mask, y_train)
train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)

val_data = TensorDataset(X_val_subject_ids, X_val_ids, X_val_mask, y_val)
val_dataloader = DataLoader(val_data, batch_size=16)

# # learning rate / hidden size loops
# 
# # Define ranges for learning rates and hidden layer sizes to try
# learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
# hidden_layer_sizes = [256, 512, 1024]
# 
# # Prepare a DataFrame to store results
# results_df = pd.DataFrame(columns=['Learning Rate', 'Hidden Layer Size', 'Train RMSE', 'Val RMSE'])
# 
# # Iterate over learning rates and hidden layer sizes
# for lr in learning_rates:
#     for hidden_size in hidden_layer_sizes:
#         # instantiate the model
#         age_model = AgePredictionModel(model)
#         
#         # Update hidden layer size
#         age_model.fc1 = nn.Linear(model.config.hidden_size, hidden_size)
#         age_model.fc2 = nn.Linear(hidden_size, hidden_size)
#         age_model.fc3 = nn.Linear(hidden_size, 1)
#         
#         # define loss and optimizer
#         criterion = nn.MSELoss()  # mean squared error for regression
#         optimizer = optim.Adam(age_model.parameters(), lr=lr)
#         
#         # Training loop with limited epochs
#         num_epochs = 3  # Adjust as needed
#         for epoch in range(num_epochs):
#             age_model.train()
#             total_loss = 0
#             total_examples = 0
#             for batch in train_dataloader:
#                 optimizer.zero_grad()
#                 subject_ids, input_ids, attention_mask, labels = batch
#                 output = age_model(input_ids, attention_mask)
#                 loss = criterion(output, labels)
#                 loss.backward()
#                 optimizer.step()
#                 total_loss += loss.item()
#                 total_examples += input_ids.shape[0]
#             train_rmse = sqrt(total_loss / total_examples)
#             
#             # Validation
#             age_model.eval()
#             val_loss = 0
#             val_examples = 0
#             with torch.no_grad():
#                 for batch in val_dataloader:
#                     subject_ids, input_ids, attention_mask, labels = batch
#                     output = age_model(input_ids, attention_mask)
#                     val_loss += criterion(output, labels).item()
#                     val_examples += input_ids.shape[0]
#                 val_rmse = sqrt(val_loss / val_examples)
#                 
#             print(f"LR: {lr}, Hidden Size: {hidden_size}, Epoch {epoch+1}/{num_epochs}: Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}")
#             
#             # Store results in DataFrame
#             row = {'Learning Rate': lr, 'Hidden Layer Size': hidden_size,
#                                             'Train RMSE': train_rmse, 'Val RMSE': val_rmse}
#             results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
# 
# # Export results to CSV
# results_df.to_csv('results.csv', index=False)

# training loop
num_epochs = 5
for epoch in range(num_epochs):
    age_model.train()
    total_loss = 0
    total_examples = 0
    train_subject_ids = []
    train_predictions = []
    train_true_labels = []
    for batch in train_dataloader:
        optimizer.zero_grad()
        subject_ids, input_ids, attention_mask, labels = batch
        output = age_model(input_ids, attention_mask)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_examples += input_ids.shape[0]
        train_subject_ids.extend(subject_ids.tolist())
        train_predictions.extend(output.tolist())
        train_true_labels.extend(labels.tolist())
    train_rmse = sqrt(total_loss / total_examples)

    # validation
    age_model.eval()
    val_loss = 0
    val_examples = 0
    val_subject_ids = []
    val_predictions = []
    val_true_labels = []
    with torch.no_grad():
        for batch in val_dataloader:
            subject_ids, input_ids, attention_mask, labels = batch
            output = age_model(input_ids, attention_mask)
            val_loss += criterion(output, labels).item()
            val_examples += input_ids.shape[0]
            val_subject_ids.extend(subject_ids.tolist())
            val_predictions.extend(output.tolist())
            val_true_labels.extend(labels.tolist())
    val_rmse = sqrt(val_loss / val_examples)

    print(f"Epoch {epoch+1}/{num_epochs}: Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}")

# print("Train Subject Ids | Train Predictions | Val True Ages")
# for sub_id, pred, true_age in zip(train_subject_ids, train_predictions, train_true_labels):
#     print(f"{sub_id} | {pred:.4f} | {true_age:.4f}")
# 
# print("Val Subject Ids | Val Predictions | Val True Ages")
# for sub_id, pred, true_age in zip(val_subject_ids, val_predictions, val_true_labels):
#     print(f"{sub_id} | {pred:.4f} | {true_age:.4f}")
# 
# # evaluation on the test set
# X_test_encoded = tokenizer(list(X_test['notes']), padding=True, truncation=True, return_tensors="pt", max_length=512)
# X_test_subject_ids = torch.tensor(X_test['subject_id'].values, dtype=int).to(device)
# X_test_ids = X_test_encoded['input_ids'].clone().to(device)
# X_test_mask = X_test_encoded['attention_mask'].clone().to(device)
# y_test = torch.tensor(y_test.values, dtype=torch.float).to(device)
# 
# test_data = TensorDataset(X_test_subject_ids, X_test_ids, X_test_mask, y_test)
# test_dataloader = DataLoader(test_data, batch_size=16)
# 
# age_model.eval()
# test_loss = 0
# test_examples = 0
# test_subject_ids = []
# test_predictions = []
# test_true_labels = []
# with torch.no_grad():
#     for batch in test_dataloader:
#         subject_ids, input_ids, attention_mask, labels = batch
#         output = age_model(input_ids, attention_mask)
#         test_loss += criterion(output, labels).item()
#         test_examples += input_ids.shape[0]
#         test_subject_ids.extend(subject_ids.tolist())
#         test_predictions.extend(output.tolist())
#         test_true_labels.extend(labels.tolist())
# test_rmse = sqrt(test_loss / test_examples)
# 
# print(f"Test RMSE: {test_rmse:.4f}")
# 
# # print("Test Subject Ids | Test Predictions | Test True Ages")
# # for sub_id, pred, true_age in zip(test_subject_ids, test_predictions, test_true_labels):
# #     print(f"{sub_id} | {pred:.4f} | {true_age:.4f}")
# 
# # save the trained model weights for future use
# #torch.save(age_model.state_dict(), 'age_prediction_model.pth')
