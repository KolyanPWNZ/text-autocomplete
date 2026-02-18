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
            attention_mask = batch['attention_mask'].to(device)

            for i in range(input_ids.size(0)):
                seq_len = attention_mask[i].sum().item()
                
                # Используем первые 75% 
                split_point = int(seq_len * 0.75)
                prefix_tokens = input_ids[i][:split_point].unsqueeze(0)
                target_tokens = input_ids[i][split_point:seq_len]

                if target_tokens.size(0) == 0:
                    continue

                prefix_attn = torch.ones_like(prefix_tokens)

                generated_ids = model.generate(
                    input_ids=prefix_tokens,
                    attention_mask=prefix_attn,
                    max_new_tokens=target_tokens.size(0),
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )

                gen_suffix = generated_ids[0, prefix_tokens.size(1):]
                pred_text = tokenizer.decode(gen_suffix, skip_special_tokens=True).strip()
                ref_text = tokenizer.decode(target_tokens, skip_special_tokens=True).strip()

                predictions.append(pred_text)
                references.append(ref_text)

    if not predictions:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    rouge_scores = rouge_metric.compute(predictions=predictions, references=references)
    return {
        'rouge1': rouge_scores['rouge1'],
        'rouge2': rouge_scores['rouge2'],
        'rougeL': rouge_scores['rougeL']
    }

def print_example_generation_transformer(model, loader, tokenizer, device, n_examples=5):
    model.eval()
    batch = next(iter(loader))
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    print(f"\nПримеры генерации трансформера (N={n_examples})")

    with torch.no_grad():
        for i in range(min(n_examples, input_ids.size(0))):
            seq_len = attention_mask[i].sum().item()
            split_point = int(seq_len * 0.75)

            prefix_tokens = input_ids[i][:split_point].unsqueeze(0)
            target_tokens = input_ids[i][split_point:seq_len]

            if target_tokens.size(0) == 0:
                continue

            # создаём корректную маску для префикса
            prefix_attn = torch.ones_like(prefix_tokens, device=device)

            gen_max_tokens = target_tokens.size(0)

            generated_ids = model.generate(
                input_ids=prefix_tokens,
                attention_mask=prefix_attn,
                max_new_tokens=gen_max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,  
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )

            gen_suffix = generated_ids[0, prefix_tokens.size(1):]
            pred_text = tokenizer.decode(gen_suffix, skip_special_tokens=True).strip()
            ref_text = tokenizer.decode(target_tokens, skip_special_tokens=True).strip()

            print(f"\nExample {i+1}:")
            print(f"Prefix:    {tokenizer.decode(prefix_tokens[0], skip_special_tokens=True).strip()}")
            print(f"Predicted: {pred_text}")
            print(f"Target:    {ref_text}")
            print("-" * 50)
