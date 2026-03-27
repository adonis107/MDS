import torch
import torch_geometric


def image_to_graph(
    image: torch.Tensor, conv2d: torch.nn.Conv2d | None = None
) -> torch_geometric.data.Data:
    """
    Converts an image tensor to a PyTorch Geometric Data object.
    COMPLETE

    Arguments:
    ----------
    image : torch.Tensor
        Image tensor of shape (C, H, W).
    conv2d : torch.nn.Conv2d, optional
        Conv2d layer to simulate, by default None
        Is used to determine the size of the receptive field.

    Returns:
    --------
    torch_geometric.data.Data
        Graph representation of the image.
    """
    # Assumptions (remove it for the bonus)
    assert image.dim() == 3, f"Expected 3D tensor, got {image.dim()}D tensor."
    if conv2d is not None:
        assert conv2d.padding[0] == conv2d.padding[1] == 2, (
            "Expected padding of 2 on both sides."
        )
        assert conv2d.kernel_size[0] == conv2d.kernel_size[1] == 5, (
            "Expected kernel size of 5x5."
        )
        assert conv2d.stride[0] == conv2d.stride[1] == 1, "Expected stride of 1."

    kernel_size = conv2d.kernel_size[0] if conv2d is not None else 5

    C, H, W = image.shape

    # Node features (each pixel is a node)
    x = image.view(C, -1).T

    # Edge index and edge attributes
    edge_list = []
    edge_attr_list = []

    half_k = kernel_size // 2
    kernel_points = [
        (i - half_k, j - half_k) for i in range(kernel_size) for j in range(kernel_size)
    ]

    for i in range(H):
        for j in range(W):
            node_idx = i * W + j
            for di, dj in kernel_points:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    neighbor_idx = ni * W + nj
                    edge_list.append((neighbor_idx, node_idx))
                    # Store relative position as edge attribute but shifted to start from (0,0)
                    edge_attr_list.append((dj + half_k, di + half_k))

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr_list = torch.tensor(edge_attr_list, dtype=torch.float)

    data = torch_geometric.data.Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr_list
    )

    return data


def graph_to_image(
    data: torch.Tensor, height: int, width: int, conv2d: torch.nn.Conv2d | None = None
) -> torch.Tensor:
    """
    Converts a graph representation of an image to an image tensor.

    Arguments:
    ----------
    data : torch.Tensor
        Graph data representation of the image.
    height : int
        Height of the image.
    width : int
        Width of the image.
    conv2d : torch.nn.Conv2d, optional
        Conv2d layer to simulate, by default None

    Returns:
    --------
    torch.Tensor
        Image tensor of shape (C, H, W).
    """
    # Assumptions (remove it for the bonus)
    assert data.dim() == 2, f"Expected 2D tensor, got {data.dim()}D tensor."
    if conv2d is not None:
        assert conv2d.padding[0] == conv2d.padding[1] == 2, (
            "Expected padding of 2 on both sides."
        )
        assert conv2d.kernel_size[0] == conv2d.kernel_size[1] == 5, (
            "Expected kernel size of 5x5."
        )
        assert conv2d.stride[0] == conv2d.stride[1] == 1, "Expected stride of 1."

    # Checking if there is no mismatch in dimensions
    N, C = data.shape
    assert N == height * width, (
        f"Data has {N} nodes, but expected {height * width} nodes."
    )

    image = data.T.view(C, height, width)

    return image


class Conv2dMessagePassing(torch_geometric.nn.MessagePassing):
    """
    A Message Passing layer that simulates a given Conv2d layer.
    """

    def __init__(self, conv2d: torch.nn.Conv2d):
        # Initialize the MessagePassing class with sum aggregation to simulate convolution over neighbors
        super().__init__(aggr="add")

        # Store the Conv2d layer and its weights
        self.conv2d = conv2d
        self.weights = conv2d.weight

    def forward(self, data):
        self.edge_index = data.edge_index

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Computes the message to be passed for each edge.
        For each edge e = (u, v) in the graph indexed by i,
        the message trough the edge e (ie from node u to node v)
        should be returned as the i-th line of the output tensor.
        (The message is phi(u, v, e) in the formalism.)
        To do this you can access the features of the source node
        in x_j[i] and the attributes of the edge in edge_attr[i].

        Arguments:
        ----------
        x_j : torch.Tensor
            The features of the souce node for each edge (of size E x in_channels).
        edge_attr : torch.Tensor
            The attributes of the edge (of size E x edge_attr_dim).

        Returns:
        --------
        torch.Tensor
            The message to be passed for each edge (of size COMPLETE)
        """

        col_idx = edge_attr[:, 0].long()
        row_idx = edge_attr[:, 1].long()

        weights = self.weights[:, :, row_idx, col_idx]
        weights = weights.permute(2, 1, 0)

        message = torch.bmm(x_j.unsqueeze(1), weights).squeeze(1)

        return message
