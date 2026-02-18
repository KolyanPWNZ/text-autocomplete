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
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        emb = self.dropout(emb)
        lstm_out, _ = self.lstm(emb)    
        logits = self.fc(lstm_out)                     
        return logits
    
    def generate(self, input_ids, max_new_tokens, tokenizer, temperature=1.0):
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  
                generated = torch.cat([generated, next_token], dim=1)
                
                if (next_token == tokenizer.eos_token_id).any():
                    break
                    
        return generated

