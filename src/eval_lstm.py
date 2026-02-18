import torch
from tqdm import tqdm
import evaluate

def calc_evaluate(model, loader, criterion, tokenizer, device, 
             max_gen_tokens=32, hide_ratio=0.25):
    
    model.eval()
    predictions, references = [], []
    total_loss = 0.0
    
    rouge_metric = evaluate.load('rouge')

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"[Evaluating]"):
            input_ids = batch['input_ids'].to(device)  
            labels = batch['labels'].to(device)        

            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()

            # считаем rouge
            for i in range(input_ids.size(0)):
                valid_mask = labels[i] != -100
                if not valid_mask.any():
                    continue
                    
                valid_tokens = input_ids[i][valid_mask]
                actual_len = valid_tokens.size(0)
                split_idx = int(actual_len * (1 - hide_ratio))
                
                prefix = valid_tokens[:split_idx].unsqueeze(0) 
                
                target_tokens = valid_tokens[split_idx:]
                ref_text = tokenizer.decode(target_tokens, skip_special_tokens=True).strip()
                
                if prefix.size(1) > 0: 
                    generated = model.generate(
                        prefix, 
                        max_new_tokens=max_gen_tokens, 
                        tokenizer=tokenizer
                    )

                    # берём только сгенерированную часть 
                    gen_suffix = generated[0, prefix.size(1):]
                    pred_text = tokenizer.decode(gen_suffix, skip_special_tokens=True).strip()
                else:
                    pred_text = ""  
                
                if ref_text:  
                    predictions.append(pred_text)
                    references.append(ref_text)
    
    if not predictions:
        return {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    
    avg_loss = total_loss / len(loader)
    rouge_scores = rouge_metric.compute(
        predictions=predictions, 
        references=references,
        use_aggregator=True
    )
    return avg_loss, rouge_scores

def print_example_generation(model, loader, tokenizer, device, n_examples=5):
    model.eval()
    batch = next(iter(loader))
    input_ids = batch['input_ids'][:n_examples].to(device)
    labels = batch['labels'][:n_examples]
    
    with torch.no_grad():
        for i in range(n_examples):
            valid_mask = labels[i] != -100
            if not valid_mask.any():
                continue
            valid_tokens = input_ids[i][valid_mask]
            split_idx = int(len(valid_tokens) * 0.75)
            prefix = valid_tokens[:split_idx].unsqueeze(0)
            target = valid_tokens[split_idx:]
            
            if prefix.size(1) > 0:
                generated = model.generate(prefix, max_new_tokens=32, tokenizer=tokenizer)
                gen_suffix = generated[0, prefix.size(1):]
                
                print(f"[Example]")
                print(f"Prefix:    {tokenizer.decode(prefix[0], skip_special_tokens=True)}")
                print(f"Predicted: {tokenizer.decode(gen_suffix, skip_special_tokens=True)}")
                print(f"Target:    {tokenizer.decode(target, skip_special_tokens=True)}")