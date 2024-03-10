import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F


# get data from file x and y
df = pd.read_csv("mlp_regression_data.csv", sep=',')
X = df["x"].values
y = df["y"].values


#-----Normalization of Data----------#
# z-score standardization around axis = 0
X_normal = (X - X.mean()) / X.std()
y_normal = (y - y.mean()) / y.std()


#------------Plot Data--------------#
def plotData(X, y, dataList, yValues):
    plt.plot(X, y, 'o' ,label="Original Data")
    plt.plot(dataList, yValues, 'r', label="non-linear regression")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc='upper left')
    plt.ylim(0,20)
    plt.xlim(0,11)
    plt.grid()
    plt.show()




#------------Data set--------------#

class myDataSet(Dataset):
    def __init__(self, X, y):
        # Load Data into PyTorch Tensors
        self.features = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.float32)

        self.features = self.features.view(-1, 1)  # reshape it into a tensor with column 1 and row -1. -1 means, that torch finds the number of rows on its own

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y
    
    def __len__(self):
        return self.labels.shape[0]
    
trainDataSet = myDataSet(X_normal, y_normal)




#------------Data Loader--------------#
trainDataLoader = DataLoader(trainDataSet, batch_size=10, shuffle=True)


#------------NN-Model--------------#
# with 3 layers (2 hidden and one output)
class MLP_regression(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linearLayer1 = torch.nn.Linear(num_features, 25)
        self.linearLayer2 = torch.nn.Linear(25, 25)
        self.linearLayer3 = torch.nn.Linear(25, 1)

    def forward(self, x):
        x = torch.relu(self.linearLayer1(x))
        x = torch.relu(self.linearLayer2(x))
        x = self.linearLayer3(x)
        return x
    


#--------Plot training Loss as a function of epoch-------#
    
def plotLoss(epoch, loss, numEpochs):
    plt.plot(epoch, loss ,label="Loss as a function of epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc='upper left')
    plt.ylim(-1, 2)
    plt.xlim(-1,numEpochs)
    plt.grid()
    plt.show()



#------------Training Loop----------------#


model = MLP_regression(num_features=1)
torch.manual_seed(123)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

numEpochs = 500


lossList = []
epochList = []
for epoch in range(numEpochs):

    model = model.train()
    
    for batch_idx, (features, classLables) in enumerate(trainDataLoader):
        # update the parameters after every batch_size

        yPredict = model(features)  # calls the forward() method on the model

        loss = F.mse_loss(yPredict, classLables.view(yPredict.shape))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    lossList.append(loss.item())
    
    epochList.append(epoch)


plotLoss(epochList, lossList, numEpochs)


#-------Evaluate Model and plot regression curve--------#

X_range = torch.arange(X.min(), X.max(), 0.01).view(-1,1)

X_range_normalize = (X_range - X.mean()) / X.std() #normalize it with original mean and std from training data


yValues = []
dataList = []
for data in X_range_normalize:
    yResult = model(data)

    yValue = (yResult.item() *  y.std()) + y.mean()  #un-normalize y value
    yValues.append(yValue)
    dataList.append((data *  X.std()) + X.mean())    #un-normalize x value

plotData(X, y, dataList, yValues)

