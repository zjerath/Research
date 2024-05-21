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
subset_size = None
num_epochs = 100
hidden_size = 32
lr = 1e-2
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

# Tokenize the train data
X_train_encoded = tokenizer(list(X_train['notes']), padding=True, truncation=True, return_tensors="pt", max_length=512)

# See the modules inside the model object
print(model._modules.keys())
# odict_keys(['embeddings', 'encoder', 'pooler'])

# Embed the first train example
embed = model._modules["embeddings"](input_ids=X_train_encoded['input_ids'][0].unsqueeze(0))

# Pass embedding into 0th layer
tmp0 = model._modules["encoder"]._modules['layer'][0](embed, attention_mask=X_train_encoded['attention_mask'][0].unsqueeze(0))
tmp0 = tmp0[0]

# Pass tmp0 into 1th layer
tmp1 = model._modules["encoder"]._modules['layer'][1](embed, attention_mask=X_train_encoded['attention_mask'][0].unsqueeze(0))
tmp1 = tmp1[0]

def pass_through_k_layers(input_ids, attention_mask, k):
    tmp = model._modules["embeddings"](input_ids=input_ids)
    assert 1 <= k <= 12
    for i in range(k):
        tmp = model._modules["encoder"]._modules['layer'][i](
            tmp,
            attention_mask=attention_mask
        )[0]

    return tmp

tmp11 = pass_through_k_layers(
    X_train_encoded['input_ids'][0].unsqueeze(0),
    X_train_encoded['attention_mask'][0].unsqueeze(0),
    12
)

pooled = model._modules["pooler"](tmp11)
normal_output = model(
    X_train_encoded['input_ids'][0].unsqueeze(0),
    X_train_encoded['attention_mask'][0].unsqueeze(0),
)

# What percentage of these values are close from the manual pass-through versus normal calculation?
print(torch.isclose(pooled, normal_output.pooler_output).float().mean())
# 0.9844

# How far off are these two calculations?
print(torch.square(pooled - normal_output.pooler_output).mean())
# 2.7048e-14

