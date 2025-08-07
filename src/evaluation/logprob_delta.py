import torch.nn.functional as F


def compute_avg_logprobs(prompt, response, model, tokenizer):
    full_text = prompt + response

    enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids.to(next(model.parameters()).device)

    prompt_len = len(
        tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
    )
    response_len = input_ids.shape[-1] - prompt_len

    outputs = model(input_ids)
    logits = outputs[0]

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    start = prompt_len
    end = start + response_len

    response_logits = shift_logits[:, start:end, :]
    response_labels = shift_labels[:, start:end]

    log_probs = F.log_softmax(response_logits, dim=-1)
    token_log_probs = log_probs.gather(2, response_labels.unsqueeze(-1)).squeeze(-1)

    avg_logprob = token_log_probs.mean().item()

    return avg_logprob


def compute_logprob_delta(prompt, preferred, rejected, model, tokenizer):
    preferred_avg = compute_avg_logprobs(prompt, preferred, model, tokenizer)
    rejected_avg = compute_avg_logprobs(prompt, rejected, model, tokenizer)
    print(
        "Preferred:",
        preferred_avg,
        "| Rejected:",
        rejected_avg,
        "| Δ =",
        preferred_avg - rejected_avg,
    )
    return preferred_avg - rejected_avg
