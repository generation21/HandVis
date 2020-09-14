import numpy as np
import glob
test = glob.glob("right_result_1\*.txt")
train =np.loadtxt(test[0])
train = np.array([train])
label = np.array([1,0,0,0,0])

for i in range(1,np.size(test)):
    addTrain = np.array([np.loadtxt(test[i])])

    # train = np.vstack((train, addTrain))
    train = np.concatenate((train, addTrain), 0)
    label = np.vstack((label, [1,0,0,0,0]))



test = glob.glob("left_result_1\*.txt")
for i in range(np.size(test)):
    addTrain = np.array([np.loadtxt(test[i])])

    train = np.concatenate((train, addTrain), 0)
    label = np.vstack((label,[1,0,0,0,0]))
print(np.shape(train))
#     ----------------------------------------------
test = glob.glob("right_result_2\*.txt")
for i in range(np.size(test)):
    addTrain = np.array([np.loadtxt(test[i])])

    train = np.concatenate((train, addTrain), 0)
    label = np.vstack((label,[0,1,0,0,0]))

test = glob.glob("left_result_2\*.txt")
for i in range(np.size(test)):
    addTrain = np.array([np.loadtxt(test[i])])

    train = np.concatenate((train, addTrain), 0)
    label = np.vstack((label,[0,1,0,0,0]))

print(np.shape(train))
#     ----------------------------------------------
test = glob.glob("right_result_3\*.txt")
for i in range(np.size(test)):
    addTrain = np.array([np.loadtxt(test[i])])

    train = np.concatenate((train, addTrain), 0)
    label = np.vstack((label,[0,0,1,0,0]))

test = glob.glob("left_result_3\*.txt")
for i in range(np.size(test)):
    addTrain = np.array([np.loadtxt(test[i])])

    train = np.concatenate((train, addTrain), 0)
    label = np.vstack((label,[0,0,1,0,0]))



print(np.shape(train))
#     ----------------------------------------------
test = glob.glob("right_result_4\*.txt")
for i in range(np.size(test)):
    addTrain = np.array([np.loadtxt(test[i])])

    train = np.concatenate((train, addTrain), 0)
    label = np.vstack((label,[0,0,0,1,0]))

test = glob.glob("left_result_4\*.txt")
for i in range(np.size(test)):
    addTrain = np.array([np.loadtxt(test[i])])

    train = np.concatenate((train, addTrain), 0)
    label = np.vstack((label,[0,0,0,1,0]))


print(np.shape(train))
#     ----------------------------------------------
test = glob.glob("right_result_5\*.txt")
for i in range(np.size(test)):
    addTrain = np.array([np.loadtxt(test[i])])

    train = np.concatenate((train, addTrain), 0)
    label = np.vstack((label,[0,0,0,0,1]))

test = glob.glob("left_result_5\*.txt")
for i in range(np.size(test)):
    addTrain = np.array([np.loadtxt(test[i])])

    train = np.concatenate((train, addTrain), 0)
    label = np.vstack((label,[0,0,0,0,1]))

#     ----------------------------------------------

print(np.shape(train))
np.savetxt('train_verstion.txt', train)
np.savetxt('label.txt', label)