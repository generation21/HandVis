# imports
import tkinter as tk
import tkinter.scrolledtext as tkst
from tkinter import Menu
from tkinter import ttk
import numpy as np
import time
import Weight
import handValue
import handValue2
import Answer9
import Answer
import queue
import torch
import torch.nn as nn
from torch.autograd import Variable
run_first = np.zeros((6,1))
operation = np.ones((6,1))
blocking = np.zeros((6,1))
NowTime = 0
softValue = 0
cnt = 0
pose_result = []
sequence_length = 5
input_size = 42
hidden_size = 40
num_layers = 2
num_classes = 7
hand_result = []
data_dim = 42
seq_length = 5
batch_size = 1
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


Q = queue.Queue(3)
State = np.zeros(5)
running = False  # Global flag



def scanning():
    if running:
        global NowTime
        global softValue
        global result
        global cnt
        global pose_result
        global hand_result
        rnn = RNN(input_size, hidden_size, num_layers, num_classes)
        rnn.load_state_dict(torch.load('rnn.pkl'))

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
                hand = np.reshape(lefthand, (1, data_dim))
            else:
                hand = np.reshape(righthand, (1, data_dim))
            if cnt == 0:
                hand_result = hand
                result = 'NotValue'
                cnt += 1
            elif cnt < seq_length:
                hand_result = np.concatenate((hand_result, hand), 0)
                result = result
                cnt += 1
            elif cnt == seq_length:
                input_data = hand_result
                input_data = np.reshape(input_data, (batch_size, seq_length, data_dim))
                result = rnn(Variable(torch.from_numpy(input_data).type_as(torch.FloatTensor())))
                result = Answer.correct(result.data.numpy())
                hand_result = np.delete(hand_result, (0), axis=0)
                cnt -= 1


            NowTime += 1
            index = int(result.split(' ')[1])
            run_first[index] += 1

            for i in range(1, 5):
                if index != i and run_first[i] > 0:
                    run_first[i] = 0

            if operation[index] == True and run_first[index] == 10 and blocking[index] == 0:
                result = result
                operation[index] = False
            else:
                result = "Action 0"

            # scrt.label(image = rd, compound = RIGHT)
        for i in range(1, 6):
            if operation[i] == False:
                blocking[i] += 1
                if blocking[i] == 20:
                    blocking[i] = 0
                    operation[i] = True
                    run_first[i] = 0
        scrt.delete(1.0, tk.END)
        if result == 'Action 7':
            result = 'others'
        display = result + '\n' + "frame :" + str(NowTime)
        # insert text in a scrolledtext
        scrt.delete(tk.END)

        scrt.insert(tk.INSERT, display)


        scrt.see(tk.END)
        scrt.update_idletasks()

    win.after(50, scanning)


# Click Start button
def start():
    """Enable scanning by setting the global flag to True."""
    global running
    global broken_count
    running = True


def stop():
    """Stop scanning by setting the global flag to False."""
    scrt.delete(1.0, tk.END)
    scrt.insert(tk.INSERT, "END\n")
    scrt.see(tk.END)
    scrt.update_idletasks()
    global running
    running = False


# Click a exit menu
def clickExit():
    win.quit()
    win.destroy()
    exit()


def initial():
    global NowTime, State, softValue, Q
    NowTime = 0
    softValue = 0
    for i in range(5):
        State[i] = 0
    for i in range(Q.qsize()):
        Q.get()


if __name__ == '__main__':
    win = tk.Tk()  # Create instance
    win.title("Hand Tracking")  # Add a title
    win.update_idletasks()

    scrt = tkst.ScrolledText(win, width=10, height=2, wrap=tk.WORD, font=("Helvetica", 50))  # Create a scrolledtext
    scrt.grid(column=0, row=0, columnspan=3)
    scrt.focus_set()  # Default focus

    action = ttk.Button(win, text="Start", command=start)  # Create a button
    action.grid(column=0, row=2)

    EndAction = ttk.Button(win, text="End", command=stop)
    EndAction.grid(column=2, row=2)

    InitialAction = ttk.Button(win, text="initial", command=initial)
    InitialAction.grid(column=1, row=2)

    # menuBar.add_cascade(label="File", menu=fileMenu)
    win.after(500, scanning)  # After 1 second, call scanning
    # win.resizable(0, 0)             # Disable resizing the GUI
    win.mainloop()  # Start GUI

