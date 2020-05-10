import torch
import torch.nn.functional as F
from torch import Tensor, nn


def gram_matrix(input: Tensor):
    batch_size, num_features, height, width = input.size()  # a=batch size(=1)
    features = input.view(
        batch_size * num_features, height * width
    )  # resise F_XL into \hat F_XL

    gram_product = torch.mm(features, features.T)  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return gram_product.div(batch_size * num_features * height * width)


class StyleLoss(nn.Module):
    def __init__(self, target_feature: Tensor):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input: Tensor) -> Tensor:
        input_gram_matrix = gram_matrix(input)
        self.loss = F.mse_loss(input_gram_matrix, self.target)
        return input


class ContentLoss(nn.Module):
    def __init__(
        self, target: Tensor,
    ):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class Normalization(nn.Module):
    def __init__(self, means: Tensor, standard_deviations: Tensor):
        super(Normalization, self).__init__()
        # Reformat the tensors to [Channels x 1 x 1] so that they can
        # directly work with image Tensor of shape [Batch x Channels x Height x Width].
        self.mean = Tensor(means).view(-1, 1, 1)
        self.std = Tensor(standard_deviations).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std
