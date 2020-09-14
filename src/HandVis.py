import socket
import numpy as np
import time
import handValue, handValue2
import Answer
import torch, torch.nn as nn
from torch.autograd import Variable
import subprocess, os, signal, glob
import threading
import requests ,json



proc = ''
NowTime = 0
cnt = 0
sequence_length = 5
input_size = 42
hidden_size = 40
num_layers = 2
num_classes = 7
hand_result = []
batch_size = 1
running = False
th = ''
thStop = False

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

rnn = RNN(input_size, hidden_size, num_layers, num_classes)
rnn.load_state_dict(torch.load('rnn.pkl'))

def main():
    global running
    global th, thStop
    proc = ""
    while 1:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(("", 5000))
        server_socket.listen(5)

        print("Waiting for Message")

        client_socket, address = server_socket.accept()

        data = client_socket.recv(512).decode()
        print(data)
        data, url = data.split(' ')

        if data == '2':
            thStop = True
            print('PID: ', proc.pid)
            if running == True:
                os.kill(proc.pid, signal.CTRL_C_EVENT)
                # proc.kill()
                running = False
            initial()

        elif data == '1':
            os.chdir("C:\\Users\\Hong\\openpose")
            if running == False:
                running = True
                str = 'C:\\Users\\Hong\\openpose\\windows\\x64\Release\\OpenPoseDemo.exe' + ' --hand --ip_camera ' + 'http://' + url + ':8080/video ' + '--write_keypoint_json ' + 'C:\\Users\\Hong\\openpose\\캡스톤\\result\\'
                proc = subprocess.Popen(str, shell=True)
                print('PID: ', proc.pid)

        if running == True:
            thStop = False
            th = threading.Thread(target=scanning)
            th.start()


        server_socket.close()


def scanning():
    global NowTime
    global cnt
    global hand_result
    global rnn, thStop
    run_first = np.zeros((8, 1))
    operation = np.ones((8, 1))
    blocking = np.zeros((8, 1))
    sequence_length = 5
    input_size = 42

    result = "NotFile"
    while(1):
        if thStop == True:
            break
        righthand = handValue.HandValue(NowTime)
        lefthand = handValue2.HandValue(NowTime)
        if righthand == 'NotValue' and lefthand == 'NotValue':
            result = "NotValue"
            NowTime = NowTime + 1
            cnt = 0
            hand_result = []
        elif righthand == "NotFile" and NowTime == 0:
            result = "NotFile"
            cnt = 0
            hand_result = []
        elif righthand == "NotFile" and NowTime > 0:
            result = result
        else:
            if righthand == 'NotValue':
                hand = np.reshape(lefthand, (1, input_size))
            else:
                hand = np.reshape(righthand, (1, input_size))
            if cnt == 0:
                hand_result = hand
                result = 'NotValue'
                cnt += 1
            elif cnt < sequence_length:
                hand_result = np.concatenate((hand_result, hand), 0)
                result = result
                cnt += 1
            elif cnt == sequence_length:
                input_data = hand_result
                # print("Input Shape: ",np.shape(input_data))
                input_data = np.reshape(input_data, (1, sequence_length, input_size))
                result = rnn(Variable(torch.from_numpy(input_data).type_as(torch.FloatTensor())))
                result = Answer.correct(result.data.numpy())
                hand_result = np.delete(hand_result, (0), axis=0)
                cnt -= 1

            NowTime += 1


        # print(result)
            # scrt.label(image = rd, compound = RIGHT)
        for i in range(1, 8):
            if operation[i] == False:
                blocking[i] += 1
                if blocking[i] == 50:
                    blocking[i] = 0
                    operation[i] = True
                    run_first[i] = 0
        # print(result, " ", NowTime ,end="")
        if result == 'Action 7':
            result = 'Action 0'
        URL = "http://165.246.243.113:9999/api/recognition"

        if result != "NotValue" and result != "NotFile":
            index = int(result.split(' ')[1])
            # print(" ", operation[index] ,' bocking', blo/85llllllllllllllllllllllll3qcking[index])
            run_first[index] += 1
            for i in range(1, 7):
                if index != i and run_first[i] > 0:
                    run_first[i] = 0

            if operation[index] == True and run_first[index] == 10 and blocking[index] == 0:
                display = result
                operation[index] = False
            else:
                display = 'Action 0'
            index = int(display.split(' ')[1])
            if index != 0:
                print(display, ' ', index)
                jaewon(URL,index)
        time.sleep(0.05)
def initial():
    global NowTime ,hand_result, cnt
    cnt = 0
    hand_result = []
    NowTime = 0
    directory = 'C:\\Users\\Hong\\openpose\\캡스톤\\result\\'
    os.chdir(directory)
    files = glob.glob('*.json')

    for filename in files:
        os.unlink(filename)
def jaewon(url, index):
    data = {'userId': 'jaewon', 'currentAction': index}
    requests.post(url, json=data)

if __name__ == "__main__":
    initial()
    main()