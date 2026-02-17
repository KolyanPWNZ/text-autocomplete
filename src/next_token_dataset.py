from torch.utils.data import Dataset
import torch


class TweetDataset(Dataset):
    def __init__(self, encodings, hiding_token=-100):
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']
        self.hiding_token = hiding_token

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        input_ids = self.input_ids[idx].clone().detach()
        attention_mask = self.attention_mask[idx].clone().detach()

        labels = input_ids.clone().detach()

        actual_length = attention_mask.sum().item()

        split_point = int(actual_length * 0.75)

        # скрываем токены контекста
        labels[:split_point] = self.hiding_token

        # помечаем паддинг 
        labels[actual_length:] = self.hiding_token

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        
