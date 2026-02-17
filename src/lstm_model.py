import torch.nn as nn
import torch


class LSTMClasssifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        out, hidden  = self.lstm(emb)
        linear_out = self.fc(out)
        return linear_out, hidden
    
    def generate(self, input_ids, max_new_tokens, tokenizer):
        self.eval()

        generated = input_ids
        
        while torch.no_grad():
            for _ in range(max_new_tokens):
                out, _ = self.forward(generated)

                next_token_logits = out[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                generated = torch.cat([generated, next_token], dim=1)

                if next_token.item() == tokenizer.eos_token_id:
                    break

        return generated

