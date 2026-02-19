import pandas as pd
import torch
from tqdm import tqdm
from src.eval_lstm import calc_evaluate
from src.save_utils import save_model


def train_model(model, train_loader, val_loader, optimizer, 
                criterion, num_epochs, tokenizer, scheduler,
                device, save_path, history_path, patience=5):
    history = []
    best_metrics = {"rouge1": -1, "rouge2": -1, "rougeL": -1, "val_loss": float('inf')}
    best_epoch = 0
    epochs_no_improve = 0
    best_model_state = None

    print(f"Начало обучения | Epochs: {num_epochs} | Patience: {patience} | Device: {device}\n")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids)

            # Сдвиг
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

        # Валидация
        val_loss, rouge_scores = calc_evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            tokenizer=tokenizer,
            device=device
        )

        scheduler.step(val_loss)

        epoch_record = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            **rouge_scores
        }

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1} | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | Rouge1: {rouge_scores["rouge1"]:.2%} | '
              f'Rouge2: {rouge_scores["rouge2"]:.2%} | RougeL: {rouge_scores["rougeL"]:.2%}\n')

        if val_loss < best_metrics['val_loss']:
            best_metrics = {
                'val_loss': val_loss,
                **rouge_scores
            } 

            best_epoch = epoch + 1
            epochs_no_improve = 0
            best_model_state = model.state_dict()

            save_model(
                model=model, tokenizer=tokenizer, save_path=save_path,
                epoch=epoch+1, metrics=rouge_scores,
                config={'batch_size': train_loader.batch_size}
            )
        else:
            epochs_no_improve += 1
            print(f"Нет улучшений по Val_loss: {epochs_no_improve}/{patience}")

        history.append(epoch_record)
        pd.DataFrame(history).to_csv(history_path, index=False)

        if epochs_no_improve >= patience:
            print(f"Early stopping на эпохе {epoch+1}")
            break

    print("\nОбучение завершено!")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Восстановлена модель с лучшим RougeL")
    print(f"История: {history_path}")
    print(f"Лучшая модель: эпоха {best_epoch}")
    print(f"Rouge:")
    for k, v in best_metrics.items():
        print(f"{k}: {v:.4f}")

    return pd.DataFrame(history)