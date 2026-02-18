import torch.nn as nn
import torch


class LSTMClasssifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, input_ids, hidden=None):
        emb = self.embedding(input_ids)
        emb = self.dropout(emb)
        lstm_out, hidden  = self.lstm(emb)    
        logits = self.fc(lstm_out)                     
        return logits, hidden
    
    def generate(self, input_ids, max_new_tokens, tokenizer, temperature=1.0):
        self.eval()
        device = input_ids.device
        current_tokens = input_ids
        
        emb = self.embedding(current_tokens)
        output, (h, c) = self.lstm(emb) 
        
        next_token_logits = self.fc(output[:, -1, :]) / temperature
        
        generated_sequence = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_sequence.append(next_token)

                if next_token.item() == tokenizer.eos_token_id:
                    break

                emb = self.embedding(next_token)
                output, (h, c) = self.lstm(emb, (h, c))
                next_token_logits = self.fc(output[:, -1, :]) / temperature

        return torch.cat([input_ids] + generated_sequence, dim=1)


