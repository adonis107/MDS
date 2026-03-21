import math

import torch
import torch.nn as nn

from detection.trainers.callbacks import EarlyStopping

class Trainer:
    def __init__(self, epochs=1000, learning_rate=1e-3, callbacks=None, device='cuda',
                 monitor='val_loss'):
        self.epochs = epochs
        self.lr = learning_rate
        self.callbacks = callbacks or []
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.monitor = monitor

    def fit(self, model, train_loader, val_loader):
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            # Training
            model.train()
            train_loss = 0
            n_ok = 0
            for batch in train_loader:
                optimizer.zero_grad()
                loss = model.training_step(batch)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                # clip_grad_norm_ returns the total gradient norm.  If any
                # gradient is NaN (e.g. from LayerNorm / softmax overflow),
                # the returned norm is NaN and the clipping multiplies all
                # gradients by NaN — corrupting every weight on .step().
                # Detect this and skip the update entirely.
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)
                if not math.isfinite(grad_norm):
                    optimizer.zero_grad()
                    continue
                optimizer.step()
                train_loss += loss.item()
                n_ok += 1

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    loss = model.training_step(batch)
                    val_loss += loss.item()

            train_loss = train_loss / max(n_ok, 1)
            val_loss /= len(val_loader)

            print(f'Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

            monitored_loss = train_loss if self.monitor == 'train_loss' else val_loss

            stop = False
            for callback in self.callbacks:
                if isinstance(callback, EarlyStopping):
                    callback(monitored_loss, model)
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
                        

def train_one_block(model, detector, train_loader, val_loader, model_type,
                    patience, epochs, lr, device, results_dir="."):
    import os

    if model_type == "transformer_ocsvm":
        # Update the existing callback path to be job-specific
        for cb in detector.trainer.callbacks:
            if isinstance(cb, EarlyStopping):
                cb.path = os.path.join(results_dir, f"{model_type}_checkpoint.pth")
                cb.reset()
        detector.trainer.fit(detector.transformer, train_loader, val_loader)
        return

    ckpt_path = os.path.join(results_dir, f"{model_type}_checkpoint.pth")
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    early_stop = EarlyStopping(patience=patience, verbose=False, path=ckpt_path)

    # PRAE: monitor train loss (which includes the mu-gate objective) because
    # the val loss is plain reconstruction and does not reflect mu convergence.
    monitor = 'val_loss'
    trainer = Trainer(epochs=epochs, learning_rate=lr,
                      callbacks=[early_stop], device=str(device),
                      monitor=monitor)
    trainer.fit(model, train_loader, val_loader)

