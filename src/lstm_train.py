import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from src.eval_lstm import calc_evaluate

from src.save_utils import save_model


def train_model(model, train_loader, val_loader, optimizer, 
                criterion, num_epochs, tokenizer, device,
                save_path, history_path, max_gen_tokens=32, patience=5):
    history = []
    best_rouge = -1
    best_epoch = 0
    epochs_no_improve = 0
    best_model_state = None
    
    print(f"Начало обучения | "
          f"Epochs: {num_epochs} | "
          f"Max gen tokens: {max_gen_tokens} | "
          f"Patience: {patience} | "
          f"Device: {device} \n")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids)
            
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Val
        val_loss, rouge_scores = calc_evaluate(
            model=model, 
            loader=val_loader, 
            criterion=criterion,
            tokenizer=tokenizer,
            device=device, 
            max_gen_tokens=max_gen_tokens
        )
        
        epoch_record = {
            'epoch': epoch + 1, 
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'rouge1': rouge_scores['rouge1'],
            'rouge2': rouge_scores['rouge2'],
            'rougeL': rouge_scores['rougeL']
        }
        
        print(f"Epoch {epoch+1} | "
                f"Train loss: {avg_train_loss:.3f} | Val loss: {val_loss:.3f} | "
                f"Rouge1: {rouge_scores['rouge1']:.3f} | "
                f"Rouge2: {rouge_scores['rouge2']:.3f} | "
                f"RougeL: {rouge_scores['rougeL']:.3f} \n")
        
        # Early stopping и сохранение лучшей модели
        if rouge_scores['rougeL'] > best_rouge:
            best_rouge = rouge_scores['rougeL']
            best_epoch = epoch + 1
            epochs_no_improve = 0

            best_model_state = model.state_dict().copy() 
            
            # Сохраняем лучшую модель
            save_model(
                model=model, tokenizer=tokenizer, save_path=save_path,
                epoch=epoch+1, metrics=rouge_scores,
                config={'batch_size': train_loader.batch_size}
            )
        else:
            epochs_no_improve += 1
            print(f"Нет улучшений: {epochs_no_improve}/{patience}")
        
        if epochs_no_improve >= patience:
            print(f"Early stopping на эпохе {epoch+1}")
            break

        history.append(epoch_record)
        pd.DataFrame(history).to_csv(history_path, index=False) # резервное сохранение истории
    

    df_history = pd.DataFrame(history)
    
    print(f"\nОбучение завершено!")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)  
        print("Восстановлена модель с лучшим RougeL")
    print(f"Лучшая модель: эпоха {best_epoch}, RougeL: {best_rouge:.3f}")
    print(f"История: {history_path}")
    
    return df_history