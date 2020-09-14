# imports

import tkinter as tk
import tkinter.scrolledtext as tkst
from tkinter import Menu
from tkinter import ttk
import numpy as np
import time
import handValue
import handValue2
import Answer
import queue
import tensorflow as tf
import requests ,json

NowTime = 0
cnt = 0
hand_result = []
operation = np.zeros((5,1))
blocking = np.zeros((5,1))

seq_length = 4
data_dim = 42
hidden_dim = 20
output_dim = 6
Q = queue.Queue(3)
State = np.zeros(5)
running = False  # Global flag

# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])

# build a LSTM network
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim,
                                           activation_fn=None)  # We use the last cell's output
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, "./tmp/model.ckpt")

def scanning():
    if running:
        global NowTime
        global result
        global cnt
        global hand_result
        global sess
        global operation
        global blocking

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
                NowTime = NowTime + 1

            if cnt == 0:
                hand_result = hand
                result = 'NotValue'
                cnt += 1
            elif cnt < seq_length:
                hand_result = np.concatenate((hand_result,hand),0)
                result = result
                cnt += 1
            elif cnt == seq_length:
                input_data = hand_result

                input_data = np.reshape(input_data, (1, seq_length, data_dim))

                result1 = sess.run(Y_pred, feed_dict={X: input_data})

                result = Answer.correct(result1)
                index = int(result.split(' ')[1])
                if operation[index - 1] == True and blocking[index - 1] == 0:
                    result = result
                    operation[index - 1] = False


                hand_result = np.delete(hand_result, (0), axis=0)
                cnt -= 1


            NowTime = NowTime + 1

        scrt.delete(1.0, tk.END)
        # display = result + '\n' + "frame :" + str(NowTime)

        display = result
        scrt.insert(tk.INSERT, display)

        scrt.see(tk.END)
        scrt.update_idletasks()
        # scrt.delete(tk.END)
        # URL = "http://165.246.228.110:9999/api/recognition"
        # data = {'value': result}
        # # res = requests.post(URL, data=json.dumps(data))
        # res = requests.post(URL, json=data)

    win.after(100, scanning)


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
    win.title("Body Tracking")  # Add a title
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

