from __future__ import print_function

import copy
import logging
from pathlib import Path
from typing import List, Tuple, Type

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from dynaconf import settings
from PIL import Image
from style_transfer.model import ContentLoss, Normalization, StyleLoss, imshow

logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.info(f"Using Device {device}")

data_path = Path("../data")


def image_loader(image_name):
    loader = transforms.Compose(
        [transforms.Resize(settings.IMAGE_SIZE), transforms.ToTensor()]
    )
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)  # add a batch dimension to fit network
    return image.to(device, torch.float)


def get_style_model_and_losses(
    cnn: Type[nn.Module], style_img: torch.Tensor, content_img: torch.Tensor,
) -> Tuple[nn.Module, List[float], List[float]]:
    cnn = copy.deepcopy(cnn)

    # VGG networks are trained on images with each channel
    # normalized by mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    # We will use them to normalize the image before sending it into the network.

    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    content_layers = ["conv_5"]
    style_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    # Make a new nn.Sequential to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[: (i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    # input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


def run_style_transfer(
    cnn,
    content_img,
    style_img,
    input_img,
    num_steps=300,
    style_weight=1000000,
    content_weight=1,
):
    """Run the style transfer."""
    logging.info("Building the style transfer model..")
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, style_img, content_img
    )
    optimizer = get_input_optimizer(input_img)

    logging.info("Optimizing..")
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for style_loss in style_losses:
                style_score += style_loss.loss
            for content_loss in content_losses:
                content_score += content_loss.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                logging.info(f"run {run[0]}:")
                logging.info(
                    "Style Loss : {:4f} Content Loss: {:4f}".format(
                        style_score.item(), content_score.item()
                    )
                )
            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img


if __name__ == "__main__":

    content_image = "krichner.jpg"
    style_image = "rembrandt.jpg"

    def imshow(tensor, title=None):
        unloader = transforms.ToPILImage()
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)  # remove the batch dimension
        image = unloader(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    content_image = image_loader(data_path / content_image)
    style_image = image_loader(data_path / style_image)
    input_img = content_image.clone()

    output = run_style_transfer(cnn, content_image, style_image, input_img,)

    plt.figure()
    imshow(output, title="Output Image")

    plt.ioff()
    plt.show()
