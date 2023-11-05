import requests

url = 'http://localhost:9595/predict'
eeg = {'AF3': 4311.79,
        'F7': 4029.74,
        'F3': 4289.23,
        'FC5': 4169.74,
        'T7': 4381.54,
        'P7': 4650.26,
        'O1': 4096.41,
        'O2': 4631.79,
        'P8': 4216.92,
        'T8': 4236.41,
        'FC6': 4194.36,
        'F4': 4287.18,
        'F8': 4617.44,
        'AF4': 4369.74}
response = requests.post(url, json=eeg).json()
print(response)
