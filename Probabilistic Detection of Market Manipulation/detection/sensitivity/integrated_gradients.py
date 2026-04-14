import torch


class IntegratedGradients:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def attribute(self, inputs, baseline=None, target_func=None, n_steps=50):
        """
        Computes feature attributions using Integrated Gradients.
        
        Args:
            inputs (torch.Tensor): Input data (Batch, Seq_Len, Features)
            baseline (torch.Tensor, optional): Baseline reference. Defaults to zeros.
            target_func (callable): Function that takes the model output and returns a scalar 
                                    (the anomaly score) to differentiate.
            n_steps (int): Number of interpolation steps.
            
        Returns:
            attributions (torch.Tensor): Importance scores same shape as inputs.
        """
        if inputs.shape[0] != 1:
            raise ValueError("Current IG implementation supports single-sample explanation (Batch=1).")

        if baseline is None:
            baseline = torch.zeros_like(inputs)

        alphas = torch.linspace(0, 1, n_steps + 1).to(inputs.device)
        
        scaled_inputs = baseline + alphas[:, None, None, None] * (inputs - baseline)
        scaled_inputs = scaled_inputs.squeeze(1) 
        scaled_inputs = scaled_inputs.detach().requires_grad_(True)

        model_output = self.model(scaled_inputs)
        
        score = target_func(model_output, scaled_inputs)
        
        grads = torch.autograd.grad(torch.sum(score), scaled_inputs)[0]
        
        avg_grads = torch.mean(grads[:-1] + grads[1:], dim=0) / 2.0
        
        attributions = (inputs - baseline) * avg_grads.unsqueeze(0)
        
        return attributions


def maximize_sigma(output, inputs):
    """
    Target function to maximize sigma. Used for the PNN model.
    """
    return output[1].sum()


def maximize_rec_error(output, inputs):
    """
    Target function to maximize reconstruction error. Used for the PRAE model.
    """
    rec = output[0] if isinstance(output, tuple) else output
    return torch.sum((inputs - rec) ** 2, dim=[1, 2])

