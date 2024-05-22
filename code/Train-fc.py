import sklearn
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


class py_data(Dataset):
    def __init__(self, x, y):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(x, pd.DataFrame):
            self.data = x.values
            self.targets = y.values
        else:
            self.data = x
            self.targets = y.squeeze()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x = torch.tensor(self.data[item], device=self.device, dtype=torch.float)
        y = torch.tensor(self.targets[item], device=self.device, dtype=torch.float)
        return x, y


class py_model(nn.Module):
    def __init__(self, dims):
        super(py_model, self).__init__()
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1], device=self.device))
            if i < len(dims) - 2:
                self.layers.append(nn.BatchNorm1d(dims[i + 1]))
                self.layers.append(nn.LeakyReLU(inplace=True))
        self.layers.to(self.device)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            input_dim = m.in_features
            stdev = 1/np.sqrt(input_dim)
            nn.init.normal_(m.weight, mean=0, std=stdev)

    def forward(self, x):
        x = x
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x


def py_train(model, data, objective, epochs=100):
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model.train().to(device=device)
    loader = DataLoader(data, batch_size=1024)
    if objective == "binary":
        loss_fn = torch.nn.BCEWithLogitsLoss()
    elif objective == "multiclass":
        loss_fn = torch.nn.CrossEntropyLoss()
    elif objective == "regression":
        loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=int(epochs / 2), gamma=0.2)
    for epoch in tqdm(range(epochs)):
        for data, target in loader:
            x = data
            predict = model(x)
            optim.zero_grad(set_to_none=True)
            loss = loss_fn(predict.squeeze(), target.type(torch.float).to(device))
            loss.backward()
            optim.step()
        #lr_scheduler.step()
        #if epoch % 10 == 0:
        print(f"train loss epoch {epoch}: ", loss.item())
    return model


def py_eval(model, data, objective="binary"):
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device=device)
    loader = DataLoader(data, batch_size=1024)
    print("len data: ", len(data))
    pred = []
    target = []
    with torch.no_grad():
        for x, y in loader:
            predict = torch.sigmoid(model(x))
            pred.extend(torch.round(predict.squeeze()).cpu().detach().tolist())
            target.extend(torch.round(y).cpu().detach().tolist())
    print(sklearn.metrics.classification_report(target, pred))
    print("confusion matrix: ")
    print(sklearn.metrics.confusion_matrix(target, pred))
    return


def train_cls(traindata, testdata, savetopath):
    df1 = pd.read_csv(traindata, testdata)
    df2 = pd.read_csv(testdata)
    print("trainset size: ", len(df1.index))
    print("testset size: ", len(df2.index))
    y_train = df1.iloc[:, -1]
    x_train = df1.iloc[:, 1:-1]
    y_test = df2.iloc[:, -1]
    x_test = df2.iloc[:, 1:-1]
    dims = len(x_train.columns)
    print("num features: ", dims)
    traindata = models.py_data(x_train, y_train)
    testdata = models.py_data(x_test, y_test)
    # FC architecture: 20 layers of width 256
    clf = models.py_model([dims] + (20*[256]) + [1])
    clf = models.py_train(clf, traindata, "binary", epochs=100)
    state_dict = clf.state_dict()
    torch.save(state_dict, savetopath)
    print("train accuracy: ")
    models.py_eval(clf, traindata, objective="binary")
    print("test accuracy: ")
    models.py_eval(clf, testdata, objective="binary")

