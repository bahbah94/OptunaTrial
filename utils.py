import torch
import torch.nn as nn
class MoaDataset:
    def __init__(self,features,targets):
        self.features = features
        self.targets = targets
    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self,item):
        return {
            "x": torch.tensor(self.features[item,:],dtype=torch.float),
            "y": torch.tensor(self.targets[item,:],dtype=torch.float),
        }


class Engine:
    def __init__(self,model,optimizer,device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    @staticmethod
    def loss_fn(targets,output):
        return nn.BCEWithLogitsLoss()(output,targets)

    def train(self,dataloader):
        self.model.train()

        final_loss = 0

        for data in dataloader:
            self.optimizer.zero_grad()
            input = data['x'].to(self.device)
            targets = data['y'].to(self.device)

            outputs = self.model(input)
            loss = self.loss_fn(targets,outputs)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        return final_loss/len(dataloader)
    def evaluate(self,dataloader):
        self.model.eval()

        final_loss = 0

        for data in dataloader:
            #self.optimizer.zero_grad()
            input = data['x'].to(self.device)
            targets = data['y'].to(self.device)

            outputs = self.model(input)
            loss = self.loss_fn(targets,outputs)
            #loss.backward()
            #self.optimizer.step()
            final_loss += loss.item()
        return final_loss/len(dataloader)


class Model(nn.Module):
    def __init__(self,nfeatures,ntargets,nlayers,hidden_size,dropout):
        super().__init__()
        layers = []
        for _ in range(nlayers):
            if len(layers) == 0:
                layers.append(nn.Linear(nfeatures,hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Dropout(dropout))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(hidden_size,hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Dropout(dropout))
                layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size,ntargets))
        self.model = nn.Sequential(*layers)

    def forward(self,x):
        return self.model(x)
