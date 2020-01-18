import pretrainedmodels
from torch import nn
import torch


def get_model(model_name: str, num_classes=1000, pretrained: bool = False):
    """

    Args:
        model_name: from the cadene list
        num_classes: number of classes for target models.
        pretrained: if model is pre-trained on imagenet.

    Returns:

    """
    # create model
    if pretrained:
        print(f"=> using pre-trained model '{model_name}'")
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained="imagenet")
    else:
        print(f"=> creating model '{model_name}'")
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=None)

    if num_classes != 1000:
        dim_feats = model.last_linear.in_features
        model.last_linear = nn.Linear(dim_feats, num_classes, bias=True)

    if hasattr(model, "avgpool"):
        model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    elif hasattr(model, "avg_pool"):
        model.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    else:
        raise NotImplementedError(f"No avgpool or avg_pool layer in the model {model_name}")

    return Net(model, model_name)


class Net(nn.Module):
    def __init__(self, model, model_name):
        super(Net, self).__init__()

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        self.features = nn.Sequential(*list(model.children())[:-1]).to(device)
        self.last_linear = list(model.children())[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.model_name = model_name

    def forward(self, x):
        x = self.features(x)

        if "densenet" in self.model_name:
            x = self.relu(x)
            x = self.pool(x)

        x = x.view(x.size()[0], -1)

        x = self.last_linear(x)
        return x
