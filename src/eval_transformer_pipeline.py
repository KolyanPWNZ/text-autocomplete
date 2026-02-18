import torch
import evaluate
from tqdm import tqdm

def evaluate_transformer(model, tokenizer, loader, device):
    model.eval()
    rouge_metric = evaluate.load('rouge')
    predictions, references = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Evaluating]"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            for i in range(len(input_ids)):
                target_mask = labels[i] != -100
                if not target_mask.any():
                    continue
                
                first_target_idx = target_mask.nonzero(as_tuple=True)[0][0].item()
                prefix_tokens = input_ids[i][:first_target_idx]
                prefix_tokens = prefix_tokens[attention_mask[i][:first_target_idx].bool()]

                target_tokens = labels[i][target_mask]
                
                ref_text = tokenizer.decode(target_tokens, skip_special_tokens=True).strip()
                if not ref_text:
                    continue

                if prefix_tokens.size(0) > 0:
                    prefix_input = prefix_tokens.unsqueeze(0).to(device) 
                    prefix_attn = torch.ones_like(prefix_input)
                    
                    gen_max_tokens = target_tokens.size(0) 
                    generated_ids = model.generate(
                        input_ids=prefix_input,
                        attention_mask=prefix_attn,
                        max_new_tokens=gen_max_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    
                    # Извлекаем ТОЛЬКО сгенерированную часть
                    gen_suffix = generated_ids[0, prefix_input.size(1):]
                    pred_text = tokenizer.decode(gen_suffix, skip_special_tokens=True).strip()
                else:
                    pred_text = ""
                
                predictions.append(pred_text)
                references.append(ref_text)

    if not predictions:
        return {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    
    rouge_scores = rouge_metric.compute(
        predictions=predictions, 
        references=references,
        use_aggregator=True
    )
    
    return {
        'rouge1': rouge_scores['rouge1'],
        'rouge2': rouge_scores['rouge2'],
        'rougeL': rouge_scores['rougeL']
    }