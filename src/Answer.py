import numpy as np

def correct(x_data):

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    y_ans = softmax(x_data)

    ans = np.argmax(y_ans)

    if ans == 0:
        return "Action 1"
    elif ans == 1:
        return "Action 2"
    elif ans == 2:
        return "Action 3"
    elif ans == 3:
        return "Action 4"
    elif ans == 4:
        return "Action 5"
    elif ans == 5:
        return "Action 6"
    elif ans == 6:
        return "Action 7"
    elif ans == 7:
        return "Action 8"
    elif ans == 8:
        return "Action 9"
    elif ans == 9:
        return "Action 10"
