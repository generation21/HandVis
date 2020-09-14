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
from PIL import Image
NowTime = 0

prev1 = "NotFile"
Q = queue.Queue(3)
State = np.zeros(10)
reseult = ""
running = False  # Global flag
W1, W2, W3, W4, b1, b2, b3, b4 = Weight.load()
filename = ""
class MyFrame(Frame):
    def __init__(self, master):
        global filename
        img = PhotoImage(file=filename)
        img = img.subsample(2)
        lbl = Label(image=img)
        lbl.image = img  # 레퍼런스 추가
        lbl.place(x=520, y=273) # 왼쪽은 120
        # lblName = Label( text="State", width=10)
        # lblName.place(x = 129, y = 270)
def scanning():
    if running:
        global NowTime
        global result
        global prev1
        global filename
        hand = handValue.HandValue(NowTime)
        if hand == 'NotValue':
            result = "No Value"
            NowTime = NowTime + 1
        elif hand == "NotFile" and NowTime == 0:
            result = "No File"
        elif hand == "NotFile" and NowTime > 0:
            result = prev1
        else:
            result = Answer9.correct(hand, W1, W2, W3, W4, b1, b2, b3, b4);
            NowTime = NowTime + 1

            # scrt.label(image = rd, compound = RIGHT)
        Q.put(result)
        if result == "Action 1":
            State[0] = State[0] + 1
        elif result == "Action 2":
            State[1] = State[1] + 1
        elif result == "Action 3":
            State[2] = State[2] + 1
        elif result == "Action 4":
            State[3] = State[3] + 1
        elif result == "Action 5":
            State[4] = State[4] + 1
        elif result == "Action 6":
            State[5] = State[5] + 1
        elif result == "Action 7":
            State[6] = State[6] + 1
        elif result == "Action 8":
            State[7] = State[7] + 1
        elif result == "Action 9":
            State[8] = State[8] + 1
        elif result == "Action 10":
            State[9] = State[9] + 1
        elif result == "No Value":
            State[10] = State[10] + 1
        else:
            State[11] = State[11] + 1

        if Q.qsize() == 3:
            softValue = 0
            MaxIndex = np.argmax(State)
            if MaxIndex == 0:
                result = "Lying"
            elif MaxIndex == 1:
                result = "Standing"
            elif MaxIndex == 2:
                result = "Sitting"
            elif MaxIndex == 3:
                result = "NotValue"
            else:
                result = "NotFile"

            remove = Q.get()
            if remove == "Lying":
                State[0] = State[0] - 1
            elif remove == "Standing":
                State[1] = State[1] - 1
            elif remove == "Sitting":
                State[2] = State[2] - 1
            elif remove == "NotValue":
                State[3] = State[3] - 1
            else:
                State[4] = State[4] - 1
        if result != 'No File':
            filename = result + ".png"
            myframe = MyFrame(win)
        prev1 = result
        scrt.delete(1.0, tk.END)
        display = result + '\n' + 'frame:' + str(NowTime)
        # insert text in a scrolledtext
        scrt.delete(tk.END)

        scrt.insert(tk.INSERT, display)


        scrt.see(tk.END)
        scrt.update_idletasks()

    win.after(150, scanning)


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

    img = tk.PhotoImage(file="7-class.png")
    # img = img.subsample(2)
    imgCtl = tk.Label(win, image=img)
    imgCtl.grid(row = 0, column = 1)


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

