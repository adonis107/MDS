from torch.utils.data import Dataset
import pandas as pd

class IndexDataset(Dataset):
    """
    Wraps a standard dataset to return (data, index).
    Required for PRAE training.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data = self.dataset[index]
        # If dataset returns (x, y), return (x, y, index)
        if isinstance(data, (list, tuple)):
             return (*data, index)
        # If dataset returns x, return (x, index)
        return data, index

    def __len__(self):
        return len(self.dataset)
    

def get_lob(path):
    """Get LOB data from a file path."""
    if path.endswith('.csv') or path.endswith('.csv.gz'):
        return pd.read_csv(path)
    elif path.endswith('.parquet'):
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format.")
    
