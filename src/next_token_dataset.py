from torch.utils.data import Dataset
import torch

class TweetDataset(Dataset):
    def __init__(self, encodings, tokenizer, hiding_token=-100):
        self.input_ids = encodings['input_ids'].clone().detach().to(dtype=torch.long)
        self.attention_mask = encodings['attention_mask'].clone().detach().to(dtype=torch.long)    
        self.hiding_token = hiding_token
        self.bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else None
        self.eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else None

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        ids = self.input_ids[idx].clone()
        mask = self.attention_mask[idx].clone()

        # Добавляем BOS/EOS токены
        if self.bos_id is not None:
            ids = torch.cat([torch.tensor([self.bos_id], dtype=torch.long), ids])
            mask = torch.cat([torch.tensor([1], dtype=torch.long), mask])

        if self.eos_id is not None:
            ids = torch.cat([ids, torch.tensor([self.eos_id], dtype=torch.long)])
            mask = torch.cat([mask, torch.tensor([1], dtype=torch.long)])

        labels = ids.clone()
        labels[mask == 0] = self.hiding_token

        return {
            "input_ids": ids,
            "attention_mask": mask,
            "labels": labels
        }

class GPTTweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, hiding_token=-100):
        self.input_ids = encodings['input_ids'].clone().detach().to(dtype=torch.long)
        self.attention_mask = encodings['attention_mask'].clone().detach().to(dtype=torch.long)
        self.hiding_token = hiding_token

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx].clone()
        attention_mask = self.attention_mask[idx].clone()
        labels = input_ids.clone()
        labels[attention_mask == 0] = self.hiding_token

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
