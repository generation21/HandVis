import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Hyper Parameters
sequence_length = 5
input_size = 42
hidden_size = 40
num_layers = 2
num_classes = 7
batch_size = 20
num_epochs = 400
learning_rate = 0.0001


class DiabetesDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt('rnn_train.txt', unpack=True)

        train12 = torch.from_numpy(np.transpose(xy)).type_as(torch.FloatTensor())
        label = np.transpose(np.loadtxt('rnn_label.txt', unpack=True))
        train = torch.chunk(train12, math.ceil(train12.size()[0] / sequence_length), dim=0)

        #        self.len = np.shape(train)[0]
        self.len = len(train)
        #        self.x_data = torch.from_numpy(train).type_as(torch.FloatTensor())
        self.x_data = train
        self.y_data = torch.from_numpy(label).type_as(torch.LongTensor())
        # print(np.shape(self.y_data))

    def __getitem__(self, index):
        img, target = self.x_data[index], self.y_data[index]
        # img = torch.unsqueeze(img,dim=0)

        return img, target

    def __len__(self):
        return self.len


# MNIST Dataset
dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=2)


# RNN Model (Many-to-One)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda()
        cell = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda()
        return hidden, cell

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1])
        return out


if __name__ == '__main__':
    rnn = RNN(input_size, hidden_size, num_layers, num_classes).cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    # Train the Model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            if batch_size > len(labels):
                continue
            # print(len(labels))
            if len(images[0]) < sequence_length:
                continue
            images = Variable(images.view(batch_size, sequence_length, input_size)).cuda()
            labels = Variable(labels).cuda()

            optimizer.zero_grad()
            outputs = rnn(images)

            loss = criterion(outputs, labels.view(-1)).cuda()

            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print('epoch: ', epoch, 'loss : ', loss)
        if epoch % 10 == 0:
            print('epoch: ', epoch)
        # Save the Model
    torch.save(rnn.state_dict(), 'rnn.pkl')
    # Test the Model
    correct = 0
    total = 0
    for images, labels in test_loader:
        if len(images[0]) < 4:
            continue
        images = Variable(images.view(1, sequence_length, input_size)).cuda()
        outputs = rnn(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        # print(predicted)
        # print(labels)
        # print(labels.view(-1))
        correct += (predicted == labels.view(-1).cuda()).sum()
        # print(labels)

    print('total :', total)
    print('correct : ', correct)
    print('Test Accuracy of the model on the 10000 test images: %f %%' % (100 * correct / total))

