import numpy as np
import json

def HandValue(time):
    try:
        directory = 'C:\\Users\\Hong\\openpose\\캡스톤\\result\\'
        num = str(time)
        num = num.zfill(12)
        keypoint = directory + num + '_keypoints.json'

        with open(keypoint) as file:
            data = json.load(file)

        hand = data["people"][0]["hand_right_keypoints"]

        handResult = []
        # Vector Normalize
        for j in range(0, np.size(hand), 3):
            # cal = np.array([hand[j], hand[j + 1]])
            # a = np.linalg.norm(cal)
            if hand[j + 2]  <= 0.10:
                return "NotValue"
            # b = [(cal[0] / a), (cal[1] / a)]
            # handResult.append(b[0])
            # handResult.append(b[1])
            handResult.append(hand[j])
            handResult.append(hand[j+1])
            # handResult.append(hand[j + 2])

        if np.sum(hand) == 0:
            return "NotValue"

        return handResult
    except FileNotFoundError as e:
        return "NotFile"
    except IndexError:
        return "NotValue"
    except ZeroDivisionError:
        return "NotValue"
    except:
        return "NotFile"