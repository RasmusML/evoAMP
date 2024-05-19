import torch


class SequentialCategorical(torch.distributions.Distribution):
    def __init__(self, logits):
        self.distribution = torch.distributions.Categorical(logits=logits)

    def log_prob(self, value):
        return self.distribution.log_prob(value).sum(-1)

    def sample(self, sample_shape=torch.Size()):
        return self.distribution.sample(sample_shape)
