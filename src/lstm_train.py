import pandas as pd
import torch
from tqdm import tqdm
from src.eval_lstm import calc_evaluate
from src.save_utils import save_model



def train_model(model, train_loader, val_loader, optimizer, 
                criterion, num_epochs, tokenizer, device,
                save_path, history_path, patience=5):
    history = []
    best_rouge = {"rouge1": -1, "rouge2": -2, "rougeL": -3}
    best_epoch = 0
    epochs_no_improve = 0
    best_model_state = None
    
    print(f"Начало обучения | "
          f"Epochs: {num_epochs} | "
          f"Patience: {patience} | "
          f"Device: {device} \n")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()            
            logits, _ = model(input_ids)
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        val_loss, rouge_scores = calc_evaluate(
            model=model, 
            loader=val_loader, 
            criterion=criterion,
            tokenizer=tokenizer,
            device=device
        )
        
        epoch_record = {
            'epoch': epoch + 1, 
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            **rouge_scores
        }

        print(f'Epoch {epoch+1} | '
              f'Train Loss: {avg_train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Val Rouge1: {rouge_scores["rouge1"]:.2%} | '
              f'Val Rouge2: {rouge_scores["rouge2"]:.2%} | '
              f'Val RougeL: {rouge_scores["rougeL"]:.2%} \n')
        
        # Сохранение по лучшей метрике RougeL
        if rouge_scores['rougeL'] > best_rouge['rougeL']:
            best_rouge = rouge_scores
            best_epoch = epoch + 1
            epochs_no_improve = 0
            
            save_model(
                model=model, tokenizer=tokenizer, save_path=save_path,
                epoch=epoch+1, metrics=rouge_scores,
                config={'batch_size': train_loader.batch_size}
            )
        else:
            epochs_no_improve += 1
            print(f"Нет улучшений RougeL: {epochs_no_improve}/{patience}")
        
        history.append(epoch_record)
        pd.DataFrame(history).to_csv(history_path, index=False)
        
        if epochs_no_improve >= patience:
            print(f"Early stopping на эпохе {epoch+1}")
            break
    
    print(f"\nОбучение завершено!")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)  
        print("Восстановлена модель с лучшим RougeL")
    print(f"История: {history_path}")
    print(f"Лучшая модель: эпоха {best_epoch}")
    print(f"Rouge:")
    for k, v in best_rouge.items():
        print(f"{k}: {v:.4f}")
    return pd.DataFrame(history)
