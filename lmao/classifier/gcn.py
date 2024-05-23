from pyexpat import _Model
from .._utils import *
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool
from ..graph.pyg import getting_loader
device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class2bin = {'Normal':0,
             'Anomaly':1}
bin2class = ['Normal', 'Anomaly']

def class_trans(data, mode=class2bin):
    return mode[data]
     
class Model:
    def __init__(self, n_feature):
        super().__init__()
        self.model = GCN(n_feature=n_feature, hidden_channels=64).to(device)
        print(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def train(self, loader):
        self.model.train()
        for data in loader:
            data.to(device)
            out = self.model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = self.criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.

    def test(self, loader):
        self.model.eval()
        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            data.to(device)
            out = self.model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.
        
class GCN(torch.nn.Module):
    def __init__(self,n_features, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(n_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x
    
    
clf = Model(n_feature=386)

def fit(model, X, y, val_X:None, val_y:None):
    group_node_attrs = model.group_node_attrs
    embeddings = model.embeddings
    train_df = pd.DataFrame({'X':X, 'y':y})
    print('Transform y from label to binary.')
    train_df['y'] = train_df['y'].progress_apply(class_trans, mode=class2bin)
    train_loader = getting_loader(df=train_df)
    if val_X or val_y :
        val_df = pd.DataFrame({'X':val_X, 'y':val_y})