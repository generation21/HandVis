import numpy as np
import glob

sequence_data = 5

Final_train = np.zeros((1,sequence_data,42))
Final_label = np.array([0])

# -------------------------------------------------------
test = glob.glob("left_result_1\*.txt")
train =np.loadtxt(test[0])
train = np.array([train])

for i in range(1,np.size(test)):
    addTrain = np.array([np.loadtxt(test[i])])
    # train = np.vstack((train, addTrain))
    train = np.concatenate((train, addTrain), 0)
test = glob.glob("right_result_1\*.txt")

for i in range(1,np.size(test)):
    addTrain = np.array([np.loadtxt(test[i])])
    # train = np.vstack((train, addTrain))
    train = np.concatenate((train, addTrain), 0)

temp_train = []
result_train = np.zeros((1,sequence_data,42))
result_label = np.array([0])

for i in range(0,np.shape(train)[0],sequence_data):
    cnt = 0
    for j in range(sequence_data):
        if i + j >= np.shape(train)[0]:
            cnt = -1
            break
        temp_train = np.concatenate((temp_train, train[i + j]), 0)
    if cnt != -1:
        result_train = np.concatenate((result_train, np.reshape(temp_train,(1,sequence_data,42))),0)
        result_label = np.vstack((result_label, [0]))
        temp_train = []

result_label = np.delete(result_label, (0), axis=0)
result_train = np.delete(result_train, (0), axis=0)
# print(np.shape(train))
print(np.shape(result_label))
Final_train = np.concatenate((Final_train, result_train), 0)
Final_label = np.vstack((Final_label, result_label))
# -------------------------------------------------------
test = glob.glob("left_result_2\*.txt")

train =np.loadtxt(test[0])
train = np.array([train])

for i in range(1,np.size(test)):
    addTrain = np.array([np.loadtxt(test[i])])
    # train = np.vstack((train, addTrain))
    train = np.concatenate((train, addTrain), 0)

test = glob.glob("right_result_2\*.txt")

for i in range(1,np.size(test)):
    addTrain = np.array([np.loadtxt(test[i])])
    # train = np.vstack((train, addTrain))
    train = np.concatenate((train, addTrain), 0)

temp_train = []
result_train = np.zeros((1,sequence_data,42))
result_label = np.array([1])

for i in range(0,np.shape(train)[0],sequence_data):
    cnt = 0
    for j in range(sequence_data):
        if i + j >= np.shape(train)[0]:
            cnt = -1
            break
        temp_train = np.concatenate((temp_train, train[i + j]), 0)
    if cnt != -1:
        result_train = np.concatenate((result_train, np.reshape(temp_train,(1,sequence_data,42))),0)
        result_label = np.vstack((result_label, [1]))
        temp_train = []
# print(np.shape(train))
print(np.shape(result_label))
result_label = np.delete(result_label, (0), axis=0)
result_train = np.delete(result_train, (0), axis=0)
Final_train = np.concatenate((Final_train, result_train), 0)
Final_label = np.vstack((Final_label, result_label))
# -------------------------------------------------------
test = glob.glob("left_result_3\*.txt")

train =np.loadtxt(test[0])
train = np.array([train])

for i in range(1,np.size(test)):
    addTrain = np.array([np.loadtxt(test[i])])
    # train = np.vstack((train, addTrain))
    train = np.concatenate((train, addTrain), 0)

test = glob.glob("right_result_3\*.txt")

for i in range(1,np.size(test)):
    addTrain = np.array([np.loadtxt(test[i])])
    # train = np.vstack((train, addTrain))
    train = np.concatenate((train, addTrain), 0)

temp_train = []
result_train = np.zeros((1,sequence_data,42))
result_label = np.array([2])

for i in range(0,np.shape(train)[0],sequence_data):
    cnt = 0
    for j in range(sequence_data):
        if i + j >= np.shape(train)[0]:
            cnt = -1
            break
        temp_train = np.concatenate((temp_train, train[i + j]), 0)
    if cnt != -1:
        result_train = np.concatenate((result_train, np.reshape(temp_train,(1,sequence_data,42))),0)
        result_label = np.vstack((result_label, [2]))
        temp_train = []
# print(np.shape(train))
print(np.shape(result_label))
result_label = np.delete(result_label, (0), axis=0)
result_train = np.delete(result_train, (0), axis=0)
Final_train = np.concatenate((Final_train, result_train), 0)
Final_label = np.vstack((Final_label, result_label))
#     ----------------------------------------------
test = glob.glob("left_result_4\*.txt")

train =np.loadtxt(test[0])
train = np.array([train])

for i in range(1,np.size(test)):
    addTrain = np.array([np.loadtxt(test[i])])
    # train = np.vstack((train, addTrain))
    train = np.concatenate((train, addTrain), 0)

test = glob.glob("right_result_4\*.txt")

for i in range(1,np.size(test)):
    addTrain = np.array([np.loadtxt(test[i])])
    # train = np.vstack((train, addTrain))
    train = np.concatenate((train, addTrain), 0)

temp_train = []
result_train = np.zeros((1,sequence_data,42))
result_label = np.array([3])

for i in range(0,np.shape(train)[0],sequence_data):
    cnt = 0
    for j in range(sequence_data):
        if i + j >= np.shape(train)[0]:
            cnt = -1
            break
        temp_train = np.concatenate((temp_train, train[i + j]), 0)
    if cnt != -1:
        result_train = np.concatenate((result_train, np.reshape(temp_train,(1,sequence_data,42))),0)
        result_label = np.vstack((result_label, [3]))
        temp_train = []
# print(np.shape(train))
print(np.shape(result_label))
result_label = np.delete(result_label, (0), axis=0)
result_train = np.delete(result_train, (0), axis=0)
Final_train = np.concatenate((Final_train, result_train), 0)
Final_label = np.vstack((Final_label, result_label))

#     ----------------------------------------------
test = glob.glob("left_result_55\*.txt")

train =np.loadtxt(test[0])
train = np.array([train])

for i in range(1,np.size(test)):
    addTrain = np.array([np.loadtxt(test[i])])
    # train = np.vstack((train, addTrain))
    train = np.concatenate((train, addTrain), 0)

test = glob.glob("right_result_55\*.txt")

for i in range(1,np.size(test)):
    addTrain = np.array([np.loadtxt(test[i])])
    # train = np.vstack((train, addTrain))
    train = np.concatenate((train, addTrain), 0)

temp_train = []
result_train = np.zeros((1,sequence_data,42))
result_label = np.array([4])

for i in range(0,np.shape(train)[0],sequence_data):
    cnt = 0
    for j in range(sequence_data):
        if i + j >= np.shape(train)[0]:
            cnt = -1
            break
        temp_train = np.concatenate((temp_train, train[i + j]), 0)
    if cnt != -1:
        result_train = np.concatenate((result_train, np.reshape(temp_train,(1,sequence_data,42))),0)
        result_label = np.vstack((result_label, [4]))
        temp_train = []
# print(np.shape(train))
print(np.shape(result_label))
result_label = np.delete(result_label, (0), axis=0)
result_train = np.delete(result_train, (0), axis=0)
Final_train = np.concatenate((Final_train, result_train), 0)
Final_label = np.vstack((Final_label, result_label))

#     ----------------------------------------------
test = glob.glob("left_result_0\*.txt")

train =np.loadtxt(test[0])
train = np.array([train])

for i in range(1,np.size(test)):
    addTrain = np.array([np.loadtxt(test[i])])
    # train = np.vstack((train, addTrain))
    train = np.concatenate((train, addTrain), 0)

test = glob.glob("right_result_0\*.txt")

for i in range(1,np.size(test)):
    addTrain = np.array([np.loadtxt(test[i])])
    # train = np.vstack((train, addTrain))
    train = np.concatenate((train, addTrain), 0)

temp_train = []
result_train = np.zeros((1,sequence_data,42))
result_label = np.array([5])

for i in range(0,np.shape(train)[0],sequence_data):
    cnt = 0
    for j in range(sequence_data):
        if i + j >= np.shape(train)[0]:
            cnt = -1
            break
        temp_train = np.concatenate((temp_train, train[i + j]), 0)
    if cnt != -1:
        result_train = np.concatenate((result_train, np.reshape(temp_train,(1,sequence_data,42))),0)
        result_label = np.vstack((result_label, [5]))
        temp_train = []
# print(np.shape(train))
print(np.shape(result_label))
result_label = np.delete(result_label, (0), axis=0)
result_train = np.delete(result_train, (0), axis=0)
Final_train = np.concatenate((Final_train, result_train), 0)
Final_label = np.vstack((Final_label, result_label))

#     ----------------------------------------------
test = glob.glob("left_result_other\*.txt")

train = np.loadtxt(test[0])
train = np.array([train])

for i in range(1, np.size(test)):
    addTrain = np.array([np.loadtxt(test[i])])
    # train = np.vstack((train, addTrain))
    train = np.concatenate((train, addTrain), 0)

test = glob.glob("right_result_other\*.txt")

for i in range(1, np.size(test)):
    addTrain = np.array([np.loadtxt(test[i])])
    # train = np.vstack((train, addTrain))
    train = np.concatenate((train, addTrain), 0)

temp_train = []
result_train = np.zeros((1, sequence_data, 42))
result_label = np.array([6])

for i in range(0, np.shape(train)[0], sequence_data):
    cnt = 0
    for j in range(sequence_data):
        if i + j >= np.shape(train)[0]:
            cnt = -1
            break
        temp_train = np.concatenate((temp_train, train[i + j]), 0)
    if cnt != -1:
        result_train = np.concatenate((result_train, np.reshape(temp_train, (1, sequence_data, 42))), 0)
        result_label = np.vstack((result_label, [6]))
        temp_train = []
# print(np.shape(train))
print(np.shape(result_label))
result_label = np.delete(result_label, (0), axis=0)
result_train = np.delete(result_train, (0), axis=0)
Final_train = np.concatenate((Final_train, result_train), 0)
Final_label = np.vstack((Final_label, result_label))
Final_train = np.delete(Final_train, (0), axis=0)
Final_label = np.delete(Final_label, (0), axis=0)
#     ----------------------------------------------

Final_train = np.reshape(Final_train, (-1,42))
print(np.shape(Final_train))
print('label shape')
print(np.shape(Final_label))
np.savetxt('rnn_train.txt', Final_train)
np.savetxt('rnn_label.txt', Final_label)