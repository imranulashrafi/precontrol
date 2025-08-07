import torch
import torch.nn.functional as F


class ActivationsGenerator:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def extract_activations(self, generated_loader, preferred_loader, rejected_loader):
        result = {}

        result["generated"] = self._process_batches(generated_loader, generate=True)
        result["preferred"] = self._process_batches(preferred_loader, generate=False)
        result["rejected"] = self._process_batches(rejected_loader, generate=False)

        return result

    def _process_batches(self, dataloader, generate=False):
        all_activations, all_masks = [], []

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            prompts = batch["prompts"]

            if generate:
                hidden_states, responses = self._extract_generated(
                    input_ids, attention_mask, prompts
                )
            else:
                hidden_states = self._extract_forwarded(
                    input_ids, attention_mask, prompts
                )

            seq_lens = [h.shape[0] for h in hidden_states]
            max_len = max(seq_lens)
            hidden_dim = hidden_states[0].shape[1]

            padded_h = torch.zeros(
                len(hidden_states), max_len, hidden_dim, device="cpu"
            )
            padded_m = torch.zeros(
                len(hidden_states), max_len, dtype=torch.int, device="cpu"
            )

            for i, h in enumerate(hidden_states):
                padded_h[i, : h.shape[0], :] = h.cpu()
                padded_m[i, : h.shape[0]] = 1

            all_activations.append(padded_h)
            all_masks.append(padded_m)

        final_activations = torch.cat(all_activations, dim=0)
        final_masks = torch.cat(all_masks, dim=0)

        if generate:
            return final_activations, final_masks, responses

        return final_activations, final_masks

    def _extract_generated(self, input_ids, attention_mask, prompts):
        responses = []

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        generated_texts = self.tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )

        for prompt, response in zip(prompts, generated_texts):
            responses.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "results": response.removeprefix(prompt),
                }
            )

        hidden = [step[-1][:, -1, :] for step in outputs.hidden_states]
        hidden = torch.stack(hidden, dim=1)
        return list(hidden[i] for i in range(hidden.size(0))), responses

    def _extract_forwarded(self, input_ids, attention_mask, prompts):
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        last_hidden = outputs.hidden_states[-1]
        batch_hidden = []

        for i in range(input_ids.size(0)):
            prompt_tokens = self.tokenizer(
                prompts[i].split("\nAssistant:")[0], return_tensors="pt"
            )["input_ids"][0]
            prompt_len = prompt_tokens.size(0)

            seq_len = attention_mask[i].sum().item()
            response_hidden = last_hidden[i, prompt_len:seq_len, :]
            batch_hidden.append(response_hidden)

        return batch_hidden
