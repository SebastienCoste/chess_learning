import torch
import torch.nn as nn
import numpy as np

"""
Spectral normalization controls the Lipschitz constant of your network, making training more stable by preventing weight matrices from amplifying gradients excessively .
 Apply this to convolutional layers that showed gradient instability.
"""
def spectral_norm_conv(module, coeff=1.0):
    # Get weight parameter
    weight = module.weight

    # Calculate spectral norm
    weight_mat = weight.view(weight.size(0), -1)
    with torch.no_grad():
        u = torch.randn(weight_mat.size(0), 1, device=weight.device)
        u.data = u.data / torch.norm(u.data, p=2)

        # Power iteration to approximate largest singular value
        for _ in range(5):
            v = torch.matmul(weight_mat.t(), u)
            v = v / torch.norm(v, p=2)
            u = torch.matmul(weight_mat, v)
            u = u / torch.norm(u, p=2)

        sigma = torch.matmul(u.t(), torch.matmul(weight_mat, v))

    # Apply spectral normalization
    weight_sn = weight / (sigma / coeff)

    return weight_sn

"""
Mixup creates new training samples by linearly interpolating between pairs of inputs and their labels . This encourages the model to behave linearly between training examples, reducing memorization and improving generalization.
"""
def mixup_data(x, y, alpha=0.2):
    """Applies mixup augmentation to the batch"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Calculates the mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def centralize_gradient(x, dim=0):
    """Gradient centralization for better training stability"""
    mean = x.mean(dim=dim, keepdim=True)
    return x - mean

def compute_grad_variance(model):
    grad_var_dict = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_var = torch.var(param.grad.data)
            grad_var_dict[name] = grad_var.item()
    return grad_var_dict