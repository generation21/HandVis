import numpy as np
import glob

seq_num = 4


Final_train = np.zeros((1,seq_num,42))
Final_label = np.array([1,0,0,0,0,0])

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
result_train = np.zeros((1,seq_num,42))
result_label = np.array([1,0,0,0,0,0])

for i in range(0,np.shape(train)[0],seq_num):
    cnt = 0
    for j in range(seq_num):
        if i + j >= np.shape(train)[0]:
            cnt = -1
            break
        temp_train = np.concatenate((temp_train, train[i + j]), 0)
    if cnt != -1:
        result_train = np.concatenate((result_train, np.reshape(temp_train,(1,seq_num,42))),0)
        result_label = np.vstack((result_label, [1,0,0,0,0,0]))
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
result_train = np.zeros((1,seq_num,42))
result_label = np.array([0,1,0,0,0,0])

for i in range(0,np.shape(train)[0],seq_num):
    cnt = 0
    for j in range(seq_num):
        if i + j >= np.shape(train)[0]:
            cnt = -1
            break
        temp_train = np.concatenate((temp_train, train[i + j]), 0)
    if cnt != -1:
        result_train = np.concatenate((result_train, np.reshape(temp_train,(1,seq_num,42))),0)
        result_label = np.vstack((result_label, [0,1,0,0,0,0]))
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
result_train = np.zeros((1,seq_num,42))
result_label = np.array([0,0,1,0,0,0])

for i in range(0,np.shape(train)[0],seq_num):
    cnt = 0
    for j in range(seq_num):
        if i + j >= np.shape(train)[0]:
            cnt = -1
            break
        temp_train = np.concatenate((temp_train, train[i + j]), 0)
    if cnt != -1:
        result_train = np.concatenate((result_train, np.reshape(temp_train,(1,seq_num,42))),0)
        result_label = np.vstack((result_label, [0,0,1,0,0,0]))
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
result_train = np.zeros((1,seq_num,42))
result_label = np.array([0,0,0,1,0,0])

for i in range(0,np.shape(train)[0],seq_num):
    cnt = 0
    for j in range(seq_num):
        if i + j >= np.shape(train)[0]:
            cnt = -1
            break
        temp_train = np.concatenate((temp_train, train[i + j]), 0)
    if cnt != -1:
        result_train = np.concatenate((result_train, np.reshape(temp_train,(1,seq_num,42))),0)
        result_label = np.vstack((result_label, [0,0,0,1,0,0]))
        temp_train = []
# print(np.shape(train))
print(np.shape(result_label))
result_label = np.delete(result_label, (0), axis=0)
result_train = np.delete(result_train, (0), axis=0)
Final_train = np.concatenate((Final_train, result_train), 0)
Final_label = np.vstack((Final_label, result_label))

#     ----------------------------------------------
test = glob.glob("left_result_5\*.txt")

train = np.loadtxt(test[0])
train = np.array([train])

for i in range(1, np.size(test)):
    addTrain = np.array([np.loadtxt(test[i])])
    # train = np.vstack((train, addTrain))
    train = np.concatenate((train, addTrain), 0)

test = glob.glob("right_result_5\*.txt")

for i in range(1, np.size(test)):
    addTrain = np.array([np.loadtxt(test[i])])
    # train = np.vstack((train, addTrain))
    train = np.concatenate((train, addTrain), 0)

temp_train = []
result_train = np.zeros((1, seq_num, 42))
result_label = np.array([0,0,0,0,1,0])

for i in range(0, np.shape(train)[0], seq_num):
    cnt = 0
    for j in range(seq_num):
        if i + j >= np.shape(train)[0]:
            cnt = -1
            break
        temp_train = np.concatenate((temp_train, train[i + j]), 0)
    if cnt != -1:
        result_train = np.concatenate((result_train, np.reshape(temp_train, (1, seq_num, 42))), 0)
        result_label = np.vstack((result_label, [0,0,0,0,1,0]))
        temp_train = []
# print(np.shape(train))
print(np.shape(result_label))
result_label = np.delete(result_label, (0), axis=0)
result_train = np.delete(result_train, (0), axis=0)
Final_train = np.concatenate((Final_train, result_train), 0)
Final_label = np.vstack((Final_label, result_label))
#     ----------------------------------------------
test = glob.glob("left_result_downward\*.txt")

train = np.loadtxt(test[0])
train = np.array([train])

for i in range(1, np.size(test)):
    addTrain = np.array([np.loadtxt(test[i])])
    # train = np.vstack((train, addTrain))
    train = np.concatenate((train, addTrain), 0)

test = glob.glob("right_result_downward\*.txt")

for i in range(1, np.size(test)):
    addTrain = np.array([np.loadtxt(test[i])])
    # train = np.vstack((train, addTrain))
    train = np.concatenate((train, addTrain), 0)

temp_train = []
result_train = np.zeros((1, seq_num, 42))
result_label = np.array([0,0,0,0,0,1])

for i in range(0, np.shape(train)[0], seq_num):
    cnt = 0
    for j in range(seq_num):
        if i + j >= np.shape(train)[0]:
            cnt = -1
            break
        temp_train = np.concatenate((temp_train, train[i + j]), 0)
    if cnt != -1:
        result_train = np.concatenate((result_train, np.reshape(temp_train, (1, seq_num, 42))), 0)
        result_label = np.vstack((result_label, [0,0,0,0,0,1]))
        temp_train = []
# print(np.shape(train))
print(np.shape(result_label))
result_label = np.delete(result_label, (0), axis=0)
result_train = np.delete(result_train, (0), axis=0)
Final_train = np.concatenate((Final_train, result_train), 0)
Final_label = np.vstack((Final_label, result_label))
# --------------------------------------------------
Final_train = np.delete(Final_train, (0), axis=0)
Final_label = np.delete(Final_label, (0), axis=0)
#     ----------------------------------------------

Final_train = np.reshape(Final_train, (-1,42))
print(np.shape(Final_train))
print('label shape')
print(np.shape(Final_label))
np.savetxt('rnn_train.txt', Final_train)
np.savetxt('rnn_label_tensorflow.txt', Final_label)