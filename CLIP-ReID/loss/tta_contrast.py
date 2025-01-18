
import torch
import torch.nn as nn

class TTACont(nn.Module):
    def __init__(self, device, k=10, temperature=2.5):
        super(TTACont, self).__init__()
        self.device = device
        self.temperature = temperature
        self.k = k
    
    def sigmoid(self, x):
        """
        Compute the sigmoid of a PyTorch tensor.

        Parameters:
        x (torch.Tensor): The input tensor.

        Returns:
        torch.Tensor: The sigmoid of the input tensor.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor.")
        return 1 / (1 + torch.exp(-x))

    # def forward(self, S): 
    #     # breakpoint()
    #     batch_size = S.shape[0] 
    #     S = self.sigmoid(S / self.temperature)
    #     rowwise_sum = S.sum(dim=1, keepdim=True)
    #     S_soft = S / rowwise_sum
    #     indices = S.argsort(dim=1)
    #     loss = 0.0
    #     for i in range(batch_size):
    #         sum = 0.0
    #         for j in range(batch_size):
    #             if indices[i,j] < self.k:
    #                 sum += S_soft[i,j]
    #         loss += - torch.log(sum)

    #     loss /= batch_size
    #     return loss

    def forward(self, S):
        #breakpoint()
        batch_size = S.shape[0]
        # Apply the sigmoid function with temperature scaling
        S = self.sigmoid(S / self.temperature)
        # Normalize row-wise to create a soft assignment
        S_soft = S / S.sum(dim=1, keepdim=True)

        # Get the sorted indices for each row
        # breakpoint()
        S_soft, _ = S_soft.sort(dim=1, descending=False)
        #breakpoint()
        # Create a mask for the top-k elements
        #top_k_mask = (indices < self.k)

        # Use the mask to sum the contributions of top-k elements for each row
        #sum_top_k = (S_soft * top_k_mask).sum(dim=1)
        sum_top_k = S_soft[:, -self.k:].sum(dim=1)
        #breakpoint()
        # Keep only positive values (avoid log(0))
        #positive_values = sum_top_k[sum_top_k > 0]

        # Compute the loss only over positive values
        # Add a small epsilon to avoid log(0) issues
        epsilon = 1e-10
        loss = -(sum_top_k * torch.log(sum_top_k + epsilon)).mean()

        return loss
