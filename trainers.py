import torch

from constants import IMAGE_KEY, LABEL_KEY


class Trainer:
    def __init__(self, model, optimizer, loss_func, train_loader, val_loader):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_func
        self.train_loader = train_loader
        self.validation_loader = val_loader

    def train_step(self):
        self.model.train()
        train_losses = []
        for batch in self.train_loader:
            x, y = batch[IMAGE_KEY], batch[LABEL_KEY]
            prediction = self.model(x)
            loss = self.loss_function(prediction.to(torch.float32), y.to(torch.float32))
            train_losses.append(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return train_losses

    def evaluation_step(self):
        validation_losses = []
        with torch.no_grad():
            self.model.eval()
            for batch in self.validation_loader:
                x, y = batch[IMAGE_KEY], batch[LABEL_KEY]
                prediction = self.model(x)
                loss = self.loss_function(
                    prediction.to(torch.float32), y.to(torch.float32)
                )
                validation_losses.append(loss)
        return validation_losses


class Trainer2D:
    def __init__(self, model, optimizer, loss_func, train_loader, val_loader):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_func
        self.train_loader = train_loader
        self.validation_loader = val_loader

    def train_step(self):
        self.model.train()
        train_losses = []
        for batch in self.train_loader:
            x, y = batch[IMAGE_KEY], batch[LABEL_KEY]
            t = torch.stack(
                [
                    self.model(sample.permute(3, 0, 1, 2)).permute(1, 2, 3, 0)
                    for sample in x
                ],
                dim=0,
            )
            loss = self.loss_function(t.to(torch.float32), y.to(torch.float32))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_losses.append(loss)
        return train_losses

    def evaluation_step(self):
        val_losses = []
        with torch.no_grad():
            self.model.eval()
            for batch in self.validation_loader:
                x, y = batch[IMAGE_KEY], batch[LABEL_KEY]
                t = torch.stack(
                    [
                        self.model(sample.permute(3, 0, 1, 2)).permute(1, 2, 3, 0)
                        for sample in x
                    ]
                )
                loss = self.loss_function(t.to(torch.float32), y.to(torch.float32))

                val_losses.append(loss)
        return val_losses
