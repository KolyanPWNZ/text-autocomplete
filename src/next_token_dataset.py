from torch.utils.data import Dataset
import torch

class TweetDataset(Dataset):
    def __init__(self, encodings, hiding_token=-100):
        self.input_ids = torch.tensor(encodings['input_ids'], dtype=torch.long)
        self.attention_mask = torch.tensor(encodings['attention_mask'], dtype=torch.long)
        self.hiding_token = hiding_token

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]

        # Сдвиг для language modeling
        labels = input_ids.clone()

        # padding не должен участвовать в loss
        labels[attention_mask == 0] = self.hiding_token

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }