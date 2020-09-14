import numpy as np
import json
cnt = 0

for i in range(1,1200):

    try:
        # directory = 'downward\\'
        directory = "0\\"
        num = str(i)


        num = num.zfill(12)


        keypoint = directory + num + '_keypoints.json'

        with open(keypoint) as file:
            data = json.load(file)
        num = str(cnt)
        # leftHand = data["people"][1]["hand_left_keypoints"]
        # if np.sum(leftHand) == 0:
        #     continue
        # leftHand = data["people"][0]["hand_left_keypoints"]
        rightHand = data["people"][0]["hand_right_keypoints"]
        # rightHand = data["people"][0]["hand_left_keypoints"]
        # if np.sum(rightHand) == 0:
        #     continue
        handResult = []
        # Vector Normalize
        for j in range(0, np.size(rightHand), 3):
            handResult.append(rightHand[j])
            handResult.append(rightHand[j + 1])
            if rightHand[j + 2] <= 0.10:
                1/0;
            # handResult.append(rightHand[j + 2])

        # directory = "left_result_"+directory
        directory = "right_result_" + directory
        np.savetxt(directory + 'save'+num+'.txt', handResult,  newline=' ', delimiter=',')
        print(i,"success")
        cnt = cnt + 1
    except:
        print(i, "_no_hand_file")

