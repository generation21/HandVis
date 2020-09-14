import requests ,json
# json_data=open("000000000000_keypoints.json").read()
#
# URL = "http://165.246.230.202:5555/recognition"
# data = json.loads(json_data)
URL = "http://165.246.230.202:5555/recognition"
data = {'value': 'Action 6'}
# res = requests.post(URL, data=json.dumps(data))
res = requests.post(URL, json=data)
print(data)