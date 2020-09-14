import requests ,json
cnt = 0
while True:
    try:
        directory = 'result\\'
        num = str(cnt)
        cnt += 1
        num = num.zfill(12)
        keypoint = directory + num + '_keypoints.json'

        with open(keypoint) as file:
            data = json.load(file)
        URL = "http://165.246.230.202:5555/recognition"
        res = requests.post(URL, data=json.dumps(data))
    except:
        data = {'outer': {'inner': 'No File'}}
        res = requests.post(URL, data=json.dumps(data))
        break
