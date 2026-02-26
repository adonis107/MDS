import torch
import torch.nn as nn

from detection.trainers.callbacks import EarlyStopping

class Trainer:
    def __init__(self, epochs=1000, learning_rate=1e-3, callbacks=None, device='cuda'):
        self.epochs = epochs
        self.lr = learning_rate
        self.callbacks = callbacks or []
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def fit(self, model, train_loader, val_loader):
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            # Training
            model.train()
            train_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                loss = model.training_step(batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    loss = model.training_step(batch)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            print(f'Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

            stop = False
            for callback in self.callbacks:
                if isinstance(callback, EarlyStopping):
                    callback(val_loss, model)
                    if callback.early_stop:
                        print("Early stopping triggered")
                        # Reload the best model checkpoint saved by EarlyStopping
                        import os
                        if os.path.exists(callback.path):
                            model.load_state_dict(torch.load(callback.path, map_location=self.device, weights_only=True))
                        stop = True
                        break
            if stop:
                break
                        
