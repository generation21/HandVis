# imports
import tkinter as tk
import tkinter.scrolledtext as tkst
from tkinter import *
from tkinter import Menu
from tkinter import ttk
import numpy as np
import time
import Weight
import handValue
import Answer9
import queue
# from PIL import Image
import requests ,json
import handValue2
NowTime = 0
import threading
prev1 = "No File"

run_first = np.zeros((6,1))
operation = np.ones((6,1))
blocking = np.zeros((6,1))

Q = queue.Queue(3)
State = np.zeros(12)
reseult = ""
running = False  # Global flag
W1, W2, W3, W4, b1, b2, b3, b4 = Weight.load()
filename = ""
def jaewon(url, index):
    data = {'userId': 'jaewon', 'currentAction': index}
    requests.post(url, json=data)
def scanning():
    if running:
        global NowTime
        global result
        global prev1
        global filename
        righthand = handValue.HandValue(NowTime)
        lefthand = handValue2.HandValue(NowTime)
        if righthand == 'NotValue' and lefthand == 'NotValue':
            result = "No Value"
            NowTime = NowTime + 1
        elif righthand == "NotFile" and NowTime == 0:
            result = "No File"
        elif righthand == "NotFile" and NowTime > 0:
            # result = prev1
            result = 'Action 0'
        else:
            if righthand == 'NotValue':
                result = Answer9.correct(lefthand,W1, W2, W3, W4, b1, b2, b3, b4 );
                NowTime = NowTime + 1
            else:
                result = Answer9.correct(righthand,W1, W2, W3, W4, b1, b2, b3, b4 );
                NowTime = NowTime + 1
            index = int(result.split(' ')[1])
            run_first[index] += 1

            for i in range(1,5):
                if index != i and run_first[i] > 0:
                    run_first[i] = 0

            if operation[index] == True and run_first[index] == 10 and blocking[index] == 0:
                result = result
                operation[index] = False
            else:
                result = "Action 0"

            # scrt.label(image = rd, compound = RIGHT)
        for i in range(1,6):
            if operation[i] == False:
                blocking[i] += 1
                if blocking[i] == 20:
                    blocking[i] = 0
                    operation[i] = True
                    run_first[i] = 0

        scrt.delete(1.0, tk.END)

        URL = "http://165.246.243.33:9999/api/recognition"
        if result != "No Value" and result != "No File":
            index = int(result.split(' ')[1])
            if index != 0 and index != 1 and index != 2 and index != 3 and index != 4:
                print(result, " ", index)

                t = threading.Thread(target=jaewon, args=(URL, index))
                t.start()

        display = result + '\n' + 'frame:' + str(NowTime)
    # insert text in a scrolledtext
        scrt.delete(tk.END)

        scrt.insert(tk.INSERT, display)


        scrt.see(tk.END)
        scrt.update_idletasks()

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


# 모든 value 초기화

def initial():
    global NowTime, State, Q
    NowTime = 0
    for i in range(10):
        State[i] = 0
    for i in range(Q.qsize()):
        Q.get()


if __name__ == '__main__':
    win = tk.Tk()  # Create instance
    win.title("Hand Tracking")  # Add a title
    win.update_idletasks()

    # img = tk.PhotoImage(file="7-class.png")
    # # img = img.subsample(2)
    # imgCtl = tk.Label(win, image=img)
    # imgCtl.grid(row=0, column=1)


    scrt = tk.Text(win, width=10, height=2, wrap=tk.WORD, font=("Helvetica", 40))  # Create a scrolledtext
    scrt.grid(column=0, row=1, columnspan=3)
    # scrt.focus_set()  # Default focus

    action = ttk.Button(win, text="Start", command=start)  # Create a button
    action.grid(column=0, row=2)

    EndAction = ttk.Button(win, text="End", command=stop)
    EndAction.grid(column=2, row=2)

    InitialAction = ttk.Button(win, text="initial", command=initial)
    InitialAction.grid(column=1, row=2)
    scanning()
    # menuBar.add_cascade(label="File", menu=fileMenu)
    # win.after(500, scanning)  # After 1 second, call scanning
    # win.resizable(0, 0)             # Disable resizing the GUI
    win.mainloop()  # Start GUI

