import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error)

device = 'mps'
PRETRAINED_MODEL = 'bert-base-uncased'
NUMERICAL_FEATURES = ['title_length',
                      'body_length',
                      'tag_past_posts',
                      'author_average_past_score',
                      'body_avg_word_length',
                      'body_lexical_diversity']


def apply_log_transformation(y):
    z = np.log10(y + 0.5)
    return z


def apply_inverse_transformation(z):
    y = 10**z - 0.5
    return y


def load_data(file_path):
    df = pd.read_csv(file_path)
    df['body'] = df['body'].fillna('')
    df['text'] = df['title'] + " " + df['body']
    df['text'] = df['text'].str.lower()
    df['target'] = apply_log_transformation(df['score'])
    df = df[['text', 'target'] + NUMERICAL_FEATURES]
    return df


def prepare_data(df, test_size=0.15, random_state=42):
    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state)
    df_train, df_val = train_test_split(
        df_train, test_size=test_size, random_state=random_state)
    return df_train, df_val, df_test


def tokenize_data(df, numerical_features, max_length=512):
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)
    encodings = tokenizer(
        list(df['text']), truncation=True, padding=True, max_length=max_length)
    labels = torch.tensor(df['target'].values, dtype=torch.float32)
    numerical_features = torch.tensor(numerical_features, dtype=torch.float32)
    return encodings, labels, numerical_features


class TextDataset(Dataset):
    def __init__(self, encodings, labels, numerical_features):
        self.input_ids = torch.tensor(encodings['input_ids'], dtype=torch.long)
        self.attention_mask = torch.tensor(
            encodings['attention_mask'], dtype=torch.long)
        self.labels = labels.clone().detach()
        self.numerical_features = numerical_features.clone().detach()

    def __getitem__(self, idx):
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx],
            'numerical_features': self.numerical_features[idx]
        }
        return item

    def __len__(self):
        return len(self.labels)


def create_data_loaders(train_dataset, val_dataset, test_dataset,
                        batch_size=64):
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader, test_loader


class RedditRegression(nn.Module):
    def __init__(self, num_numerical_features):
        super(RedditRegression, self).__init__()
        self.bert = BertModel.from_pretrained(PRETRAINED_MODEL)
        self.num_numerical_features = num_numerical_features
        self.regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size +
                      self.num_numerical_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, numerical_features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        combined_output = torch.cat((pooled_output, numerical_features), dim=1)
        prediction = self.regressor(combined_output)
        return prediction


def process_batch(model, batch, loss_fn, model_mode='train'):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    numerical_features = batch['numerical_features'].to(device)

    if model_mode == 'train':
        predictions = model(input_ids, attention_mask, numerical_features)
    if model_mode == 'evaluate':
        with torch.no_grad():
            predictions = model(input_ids, attention_mask, numerical_features)

    loss = loss_fn(predictions.squeeze(), labels)

    return labels, predictions, loss


def train(model, data_loader, loss_fn, optimizer,
          original_scale=False):
    model.train()
    train_losses = []
    train_maes = []
    train_mapes = []
    with tqdm(data_loader, desc='Training', leave=False) as t:
        for batch in t:
            labels, predictions, loss = process_batch(
                model, batch, loss_fn, model_mode='train')
            train_losses.append(loss.item())

            labels = labels.cpu().detach().numpy()
            predictions = predictions.cpu().detach().numpy()
            if original_scale:
                predictions = apply_inverse_transformation(
                    predictions)
                labels = apply_inverse_transformation(
                    labels)

            train_mae = mean_absolute_error(labels, predictions)
            train_maes.append(train_mae)
            train_mape = mean_absolute_percentage_error(labels, predictions)
            train_mapes.append(train_mape)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(
                {'Loss': loss.item(), 'MAE': train_mae,
                 'MAPE': train_mape})

    avg_loss = np.mean(train_losses)
    avg_mae = np.mean(train_maes)
    avg_mape = np.mean(train_mapes)

    return avg_loss, avg_mae, avg_mape


def evaluate(model, data_loader, loss_fn, original_scale=False):
    model.eval()
    losses = []
    maes = []
    mapes = []
    eval_yhats = []
    eval_labels = []

    with tqdm(data_loader, desc='Evaluation', leave=False) as t:
        for batch in t:
            labels, predictions, loss = process_batch(
                model, batch, loss_fn, model_mode='evaluate')
            losses.append(loss.item())

            labels = labels.cpu().detach().numpy()
            predictions = predictions.cpu().detach().numpy()
            if original_scale:
                predictions = apply_inverse_transformation(
                    predictions)
                labels = apply_inverse_transformation(
                    labels)

            mae = mean_absolute_error(labels, predictions)
            maes.append(mae)
            mape = mean_absolute_percentage_error(labels, predictions)
            mapes.append(mape)
            eval_yhats.append(predictions)
            eval_labels.append(labels)

            t.set_postfix({'Loss': loss.item(), 'MAE': mae,
                           'MAPE': mape})

    avg_loss = np.mean(losses)
    avg_mae = np.mean(maes)
    avg_mape = np.mean(mapes)

    return avg_loss, avg_mae, avg_mape, eval_labels, eval_yhats
