import torch
import torch.nn.functional as F

from src.models.value_model.factory_value_model import BaseValueFunctionModule


class ValueFunctionModuleMarginRegularizer(BaseValueFunctionModule):
    def shared_step(self, batch, stage):
        activations = batch["generated_activations"]
        masks = batch["generated_masks"]
        responses = batch["generated_responses"]
        rewards = batch["rewards"]

        pref_activations = batch["preferred_activations"]
        pref_masks = batch["preferred_masks"]

        rej_activations = batch["rejected_activations"]
        rej_masks = batch["rejected_masks"]

        batch_size, seq_len, hidden_dim = activations.shape

        predictions = self(activations.view(-1, hidden_dim)).view(
            batch_size, seq_len, -1
        )
        predictions_pref = self(pref_activations.view(-1, hidden_dim)).view(
            batch_size, seq_len, -1
        )
        predictions_rej = self(rej_activations.view(-1, hidden_dim)).view(
            batch_size, seq_len, -1
        )

        valid_mask = masks[:, :-1] * masks[:, 1:]
        valid_preds = predictions[:, :-1][valid_mask.bool()]
        next_valid_preds = predictions[:, 1:][valid_mask.bool()]

        pairwise_loss = F.mse_loss(valid_preds, next_valid_preds, reduction="mean")

        lengths = masks.long().sum(dim=1)
        last_indices = (lengths - 1).clamp(min=0)
        batch_indices = torch.arange(batch_size, device=self.device)

        final_preds = predictions[batch_indices, last_indices]

        final_loss = F.mse_loss(final_preds, rewards, reduction="mean")

        lengths_pref = pref_masks.long().sum(dim=1)
        last_indices_pref = (lengths_pref - 1).clamp(min=0)

        lengths_rej = rej_masks.long().sum(dim=1)
        last_indices_rej = (lengths_rej - 1).clamp(min=0)

        final_preds_pref = predictions_pref[batch_indices, last_indices_pref]
        final_preds_rej = predictions_rej[batch_indices, last_indices_rej]

        margin_loss = -F.logsigmoid(final_preds_pref - final_preds_rej).mean()

        regularizer_loss = F.mse_loss(final_preds, final_preds_pref, reduction="mean")

        total_loss = pairwise_loss + final_loss + margin_loss + regularizer_loss

        is_train = stage == "train"
        is_val = stage == "val"

        self.log(
            f"{stage}_total_loss",
            total_loss,
            on_step=is_train,
            on_epoch=is_val,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}_pairwise_loss",
            pairwise_loss,
            on_step=is_train,
            on_epoch=is_val,
            logger=True,
        )
        self.log(
            f"{stage}_reward_loss",
            final_loss,
            on_step=is_train,
            on_epoch=is_val,
            logger=True,
        )
        self.log(
            f"{stage}_margin_loss",
            margin_loss,
            on_step=is_train,
            on_epoch=is_val,
            logger=True,
        )

        self.log(
            f"{stage}_regularizer_loss",
            regularizer_loss,
            on_step=is_train,
            on_epoch=is_val,
            logger=True,
        )

        return total_loss
