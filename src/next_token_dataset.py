from torch.utils.data import Dataset
import torch

class TweetDataset(Dataset):
    def __init__(self, encodings, hiding_token=-100):
        # Преобразуем всё в тензоры заранее для скорости
        self.input_ids = torch.tensor(encodings['input_ids'], dtype=torch.long)
        self.attention_mask = torch.tensor(encodings['attention_mask'], dtype=torch.long)
        self.hiding_token = hiding_token

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        input_ids = self.input_ids[idx].clone()
        attention_mask = self.attention_mask[idx].clone()
        labels = input_ids.clone()

        # Находим реальную длину (без учета padding)
        actual_length = attention_mask.sum().item()
        
        # Точка разделения: 75% текста — контекст, 25% — ответ
        split_point = int(actual_length * 0.75)

        # Маскируем токены, которые модель НЕ должна предсказывать (префикс и паддинг)
        labels[:split_point] = self.hiding_token
        labels[actual_length:] = self.hiding_token

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
