import torch


class SequentialCategorical(torch.distributions.Distribution):
    def __init__(self, logits: torch.Tensor, pad_token_id: int = None):
        self.distribution = torch.distributions.Categorical(logits=logits)
        self.pad_token_id = pad_token_id

    def log_prob(self, value: torch.Tensor):
        log_prob = self.distribution.log_prob(value)

        if self.pad_token_id is None:
            return log_prob.sum(-1)

        mask = value != self.pad_token_id
        masked_log_prob = log_prob * mask.float()
        return masked_log_prob.sum(-1)

    def sample(self, sample_shape: torch.Size = torch.Size()):
        return self.distribution.sample(sample_shape)
