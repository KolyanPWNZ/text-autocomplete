import torch
from tqdm import tqdm
import evaluate

def calc_evaluate(model, loader, criterion, tokenizer, device):
    model.eval()
    predictions, references = [], []
    total_loss = 0.0
    
    rouge_metric = evaluate.load('rouge')

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Evaluating]"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits, _ = model(input_ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss += loss.item()

            for i in range(input_ids.size(0)):
                target_mask = labels[i] != -100 
                
                if not target_mask.any():
                    continue
                
                target_indices = target_mask.nonzero(as_tuple=True)[0]
                first_target_idx = target_indices[0].item()
                last_target_idx = target_indices[-1].item()
                
                prefix_mask = torch.zeros_like(attention_mask[i], dtype=torch.bool)
                prefix_mask[:first_target_idx] = attention_mask[i][:first_target_idx].bool()
                
                prefix = input_ids[i][prefix_mask].unsqueeze(0)
                
                target_tokens = labels[i][target_mask]
                ref_text = tokenizer.decode(target_tokens, skip_special_tokens=True).strip()
                
                if prefix.size(1) > 0 and ref_text:
                    gen_len = target_tokens.size(0)
                    
                    generated = model.generate(
                        prefix, 
                        max_new_tokens=gen_len, 
                        tokenizer=tokenizer
                    )

                    gen_suffix = generated[0, prefix.size(1):]
                    pred_text = tokenizer.decode(gen_suffix, skip_special_tokens=True).strip()
                    
                    predictions.append(pred_text)
                    references.append(ref_text)
    
    avg_loss = total_loss / len(loader)
    
    if not predictions:
        return avg_loss, {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    
    rouge_scores = rouge_metric.compute(
        predictions=predictions, 
        references=references,
    )
    
    return avg_loss, {
        'rouge1': rouge_scores['rouge1'],
        'rouge2': rouge_scores['rouge2'],
        'rougeL': rouge_scores['rougeL']
    }

def print_example_generation(model, loader, tokenizer, device, n_examples=5):
    model.eval()
    batch = next(iter(loader))
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'] 
    
    print(f"\nПримеры генерации (N={n_examples})")
    
    with torch.no_grad():
        # Ограничиваем цикл количеством примеров или размером батча
        for i in range(min(n_examples, input_ids.size(0))):
            target_mask = labels[i] != -100
            if not target_mask.any():
                continue
            
            target_indices = target_mask.nonzero(as_tuple=True)[0]
            first_target_idx = target_indices[0].item()
            
            prefix_mask = batch['attention_mask'][i].bool()
            prefix_mask[first_target_idx:] = False # Оставляем только часть до таргета
            
            prefix = input_ids[i][prefix_mask].unsqueeze(0)
            target_tokens = labels[i][target_mask]
            
            gen_len = target_tokens.size(0)
            
            if prefix.size(1) > 0:
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
