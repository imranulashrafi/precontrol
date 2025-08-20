from src.models.value_model.factory_value_model import BaseValueFunctionModule


class ValueFunctionModule(BaseValueFunctionModule):
    def shared_step(self, batch, stage):
        activations = batch["generated_activations"]
        masks = batch["generated_masks"]
        responses = batch["generated_responses"]
        rewards = batch["rewards"]

        batch_size, seq_len, hidden_dim = activations.shape
        predictions = self(activations.view(-1, hidden_dim)).view(
            batch_size, seq_len, -1
        )

        valid_mask = masks[:, :-1] * masks[:, 1:]
        valid_preds = predictions[:, :-1][valid_mask.bool()]
        next_valid_preds = predictions[:, 1:][valid_mask.bool()]
        pairwise_loss = F.mse_loss(valid_preds, next_valid_preds, reduction="sum")

        last_indices = masks.float().argmax(dim=1, keepdim=True)
        last_indices[masks.sum(dim=1) == 0] = -1
        batch_indices = torch.arange(batch_size, device=self.device)
        final_preds = predictions[batch_indices, last_indices.squeeze()]
        final_loss = F.mse_loss(final_preds, rewards, reduction="sum")

        total_loss = (pairwise_loss + final_loss) / batch_size

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
            pairwise_loss / batch_size,
            on_step=is_train,
            on_epoch=is_val,
            logger=True,
        )
        self.log(
            f"{stage}_reward_loss",
            final_loss / batch_size,
            on_step=is_train,
            on_epoch=is_val,
            logger=True,
        )

        return total_loss
