import torch
from tqdm import tqdm
import evaluate


def calc_evaluate(model, loader, criterion, tokenizer, device):
    model.eval()
    rouge_metric = evaluate.load('rouge')
    predictions, references = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Evaluating]"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits, _ = model(input_ids)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            total_loss += loss.item()

            # Генерация последних 25% 
            for i in range(input_ids.size(0)):
                seq_len = attention_mask[i].sum().item()
                split_point = int(seq_len * 0.75)

                prefix = input_ids[i][:split_point].unsqueeze(0)
                target_tokens = input_ids[i][split_point:seq_len]

                if target_tokens.size(0) == 0:
                    continue

                generated = model.generate(
                    prefix,
                    max_new_tokens=target_tokens.size(0),
                    tokenizer=tokenizer
                )

                gen_suffix = generated[0, prefix.size(1):]

                pred_text = tokenizer.decode(gen_suffix, skip_special_tokens=True).strip()
                ref_text = tokenizer.decode(target_tokens, skip_special_tokens=True).strip()

                if pred_text and ref_text:
                    predictions.append(pred_text)
                    references.append(ref_text)

    avg_loss = total_loss / len(loader)

    if not predictions:
        return avg_loss, {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    rouge_scores = rouge_metric.compute(predictions=predictions, references=references)

    return avg_loss, {
        "rouge1": rouge_scores['rouge1'],
        "rouge2": rouge_scores['rouge2'],
        "rougeL": rouge_scores['rougeL']
    }

def print_example_generation_lstm(model, loader, tokenizer, device, n_examples=5):
    model.eval()
    batch = next(iter(loader))
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels']

    print(f"\nПримеры генерации LSTM (N={n_examples})")

    with torch.no_grad():
        for i in range(min(n_examples, input_ids.size(0))):
            seq_len = batch['attention_mask'][i].sum().item()
            split_point = int(seq_len * 0.75)

            prefix = input_ids[i][:split_point].unsqueeze(0)
            target_tokens = input_ids[i][split_point:seq_len]

            if target_tokens.size(0) == 0:
                continue

            gen_len = target_tokens.size(0)

            generated = model.generate(
                prefix,
                max_new_tokens=gen_len,
                tokenizer=tokenizer
            )
            gen_suffix = generated[0, prefix.size(1):]

            print(f"\nExample {i+1}:")
            print(f"Prefix:    {tokenizer.decode(prefix[0], skip_special_tokens=True).strip()}")
            print(f"Predicted: {tokenizer.decode(gen_suffix, skip_special_tokens=True).strip()}")
            print(f"Target:    {tokenizer.decode(target_tokens, skip_special_tokens=True).strip()}")
            print("-" * 50)
