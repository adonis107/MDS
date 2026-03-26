import torch
import torch.nn as nn
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score
import numpy as np
from torch_geometric.nn import GATConv


# Define model (in your class_model_gnn.py) according to the architecture given in the paper
class StudentModel(nn.Module):
    """
    This model implements a 3-layer Graph Attention Network (GAT) with a skip
    connection between the first and second hidden layers.

    Parameters
    ----------
    input_size : int, optional
        Dimensionality of input features. Defaults to `n_features`.
    hidden_size : int, optional
        Dimensionality of the hidden layers (per head). Defaults to 256.
    output_size : int, optional
        Dimensionality of the output classes. Defaults to 121.
    heads_1 : int, optional
        Number of attention heads for the first layer. Defaults to 4.
    heads_2 : int, optional
        Number of attention heads for the second layer. Defaults to 4.
    heads_3 : int, optional
        Number of attention heads for the output layer. Defaults to 6.
    """

    def __init__(
        self,
        input_size=50,
        hidden_size=350,
        output_size=121,
        heads_1=4,
        heads_2=4,
        heads_3=6,
    ):
        super().__init__()

        self.elu = nn.ELU()

        # Layer 1
        self.conv1 = GATConv(
            input_size,
            hidden_size,
            heads=heads_1,
            concat=True,
        )

        # Layer 2
        self.conv2 = GATConv(
            hidden_size * heads_1,
            hidden_size,
            heads=heads_2,
            concat=True,
        )

        self.lin_skip = nn.Linear(
            hidden_size * heads_1,
            hidden_size * heads_2,
            bias=False,
        )

        # Final layer (Classifier)
        self.conv3 = GATConv(
            hidden_size * heads_2,
            output_size,
            heads=heads_3,
            concat=False,
        )

    def forward(self, x, edge_index):
        """
        Performs the forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape `(num_nodes, input_size)`.
        edge_index : torch.Tensor
            Graph connectivity in COO format of shape `(2, num_edges)`.

        Returns
        -------
        torch.Tensor
            The output logits of shape `(num_nodes, output_size)`.
        """
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.elu(x)

        # Layer 2
        x_in = x
        x = self.conv2(x, edge_index)
        x = self.elu(x)

        # Skip connection
        x = x + self.lin_skip(x_in)

        # Final layer (logits)
        x = self.conv3(x, edge_index)

        return x


# Evaluation function (from the notebook)
def evaluate(model, device, dataloader):
    score_list_batch = []

    model.eval()
    for i, batch in enumerate(dataloader):
        batch = batch.to(device)
        output = model(batch.x, batch.edge_index)
        predict = np.where(output.detach().cpu().numpy() >= 0, 1, 0)
        score = f1_score(batch.y.cpu().numpy(), predict, average="micro")
        score_list_batch.append(score)

    return np.array(score_list_batch).mean()


# Load datasets
BATCH_SIZE = 2
# Val Dataset
val_dataset = PPI(root="", split="val")
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
# Test Dataset
test_dataset = PPI(root="", split="test")
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

if __name__ == "__main__":
    ### This is the part we will run in the inference to grade your model
    ## Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StudentModel()  # !  Important : No argument
    ## Load the saved model such that we can load it also on cpu
    state_dict = torch.load("model.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully")

    # Inference on validation and test datasets
    val_scores = evaluate(model, device, val_dataloader)
    print("Final GAT Model : F1-Score on the val set: {:.4f}".format(val_scores))

    test_score = evaluate(model, device, test_dataloader)
    print("Final GAT Model : F1-Score on the test set: {:.4f}".format(test_score))
