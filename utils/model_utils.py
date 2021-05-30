import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(model_path, model_class, input_channels=4, num_classes=1):
    """
    Loads a Pytorch model on GPU if possible.

    :param model_path: model parameters dictionary .pth filename
    :type model_path: str

    :param model_class: PyTorch model class
    :type model_class: torch.nn.Module

    :param input_channels: Input channel number
    :type input_channels: int

    :param num_classes: Number of classes on output
    :type num_classes: int

    :return: pretrained model
    :rtype: torch.nn.Module
    """

    model = model_class(num_classes=num_classes, input_channels=input_channels)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    return model


def run_model(input_layer, model, dataset):
    """
    Runs an already loaded PyTorch model for evaluation.

    :param input_layer: model input
    :type input_layer: torch.autograd.Variable

    :param model: PyTorch model
    :type model: torch.nn.Module

    :param dataset: "roof": Roof dataset | "income": Income Level dataset
    :type dataset: str

    :return: model output
    :rtype: torch.autograd.Variable
    """

    model.eval()

    with torch.set_grad_enabled(False):
        if dataset == "roof":
            response = torch.sigmoid(model(input_layer))
        else:
            response = torch.exp(model(input_layer))

    return response
