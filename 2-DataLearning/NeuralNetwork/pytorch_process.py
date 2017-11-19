import matplotlib.pyplot as plt
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from data_input import prepare_data
from sklearn.metrics import r2_score, mean_squared_error


csv_file = 'movie_metadata_processed.csv'
if len(sys.argv) >= 2:
    csv_file = sys.argv[1]
trainX, trainY, testX, testY = prepare_data(os.path.join('InputData', csv_file))
model = nn.Sequential(nn.Linear(len(trainX[0]), 100, bias=False),
                      nn.BatchNorm1d(100),
                      nn.ReLU(True),
                      nn.Linear(100, 50, bias=False),
                      nn.BatchNorm1d(50),
                      nn.ReLU(True),
                      nn.Linear(50,1),
                      nn.Tanh())
model.cuda()
trainX = Variable(torch.from_numpy(trainX)).cuda().float()
trainY = Variable(torch.from_numpy(trainY)).cuda().float()
testX = Variable(torch.from_numpy(testX)).cuda().float()

optimizer = optim.Adam(model.parameters())
cost_function = nn.MSELoss()
cost_array = []
model.train()
for i in range (20000):
    predict = model(trainX) * 2 + 7
    cost = cost_function(predict, trainY)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    cost_array.append(cost.data[0])
    if i%1000 == 0:
        print(i, cost.data[0])

model.eval()
testLabel = model(testX) * 2 + 7
testLabel = testLabel.data.cpu().numpy()
print(r2_score(testY, testLabel), np.sqrt(mean_squared_error(testY, testLabel)))
plt.figure()
plt.plot(cost_array[5000:], label='test cost')
plt.xlabel('epoch')
plt.ylabel('mean square error')
plt.savefig('test_cost_per_epoch.png')
plt.show()