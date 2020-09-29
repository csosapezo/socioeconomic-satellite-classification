import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(model_path, model_class):
    """
    Loads a Pytorch model on GPU if possible.

    :param model_path: model parameters dictionary .pth filename
    :type model_path: str

    :param model_class: PyTorch model class
    :type model_class: torch.nn.Module


    :return: pretrained model
    :rtype: torch.nn.Module
    """

    model = model_class()
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    return model


def run_model(input_layer, model):
    """
    Runs an already loaded PyTorch model for evaluation.

    :param input_layer: model input
    :type input_layer: torch.autograd.Variable

    :param model: PyTorch model
    :type model: torch.nn.Module


    :return: model output
    :rtype: torch.autograd.Variable
    """

    model.eval()

    with torch.set_grad_enabled(False):
        response = model(input_layer)

    return response
