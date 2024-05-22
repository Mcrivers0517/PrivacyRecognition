from torch.utils.data import Dataset
import os
import pandas as pd
import torch
import yaml

class PrivacyDataset(Dataset):
    def __init__(self, data_dir, label_dir, tokenizer, max_len, category_map_path):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.category_map = self.load_category_map(category_map_path)
        self.data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.txt')]

    def load_category_map(self, filepath):
        with open(filepath, 'r') as file:
            data = yaml.safe_load(file)
        return data['categories']

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_path = self.data_files[idx]
        label_path = os.path.join(self.label_dir, os.path.basename(data_path).replace('.txt', '.csv'))

        with open(data_path, 'r', encoding='utf-8') as file:
            text = file.read()

        labels_df = pd.read_csv(label_path)
        labels = [0] * len(text)

        for _, row in labels_df.iterrows():
            category_id = self.category_map[row['Category']]
            for i in range(row['Pos_b'], row['Pos_e'] + 1):
                if i < len(labels):
                    labels[i] = category_id

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        label_ids = [0] * self.max_len
        label_ids[1:min(len(labels) + 1, self.max_len - 1)] = labels[:self.max_len - 2]

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }