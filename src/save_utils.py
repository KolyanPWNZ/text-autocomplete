import torch
from pathlib import Path


def save_model(model, tokenizer, save_path, epoch, metrics, config=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': model.fc.out_features,
            'embedding_dim': model.embedding.embedding_dim,
            'hidden_dim': model.lstm.hidden_size,
            'n_layers': model.lstm.num_layers,
            'dropout': model.dropout.p if hasattr(model.dropout, 'p') else 0.0
        },
        'metrics': metrics,
        'config': config or {},
        'tokenizer_class': tokenizer.__class__.__name__,
    }
    
    torch.save(checkpoint, save_path)

def load_model(load_path, device='cpu'):
    checkpoint = torch.load(load_path, map_location=device, weights_only=False)
    
    config = checkpoint['model_config']
    from src.lstm_model import LSTMClassifier  
    
    model = LSTMClassifier(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        n_layers=config['n_layers'],
        dropout=config['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint.get('tokenizer_class'), checkpoint.get('config'), checkpoint.get('metrics')