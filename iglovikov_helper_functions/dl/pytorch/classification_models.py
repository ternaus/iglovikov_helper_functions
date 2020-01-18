import pretrainedmodels
from torch import nn


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

    if model_name == "resnet50":
        model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    elif model_name in ["se_resnext50_32x4d", "se_resnext101_32x4d"]:
        model.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    else:
        raise NotImplementedError(f"Average pool for is not added to {model_name}")

    return Net(model)


class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.l1 = nn.Sequential(*list(model.children())[:-1]).to("cuda:0")
        self.last = list(model.children())[-1]

    def forward(self, x):
        x = self.l1(x)
        x = x.view(x.size()[0], -1)
        x = self.last(x)
        return x
