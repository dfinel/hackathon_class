import requests
import json

# URL of your Flask app
url = 'http://127.0.0.1:5000/process_test_data'

# JSON data to send
data = {
    'Medu': 4,
    'Fedu': 3,
    'traveltime': 2,
    'studytime': 2,
    'freetime': 3,
    'health': 4,
    'sex': 'F',
    'address': 'U',
    'famsize': 'GT3',
    'Pstatus': 'A',
    'paid': 'yes',
    'activities': 'yes',
    'nursery': 'yes',
    'internet': 'yes',
    'mjob': 'teacher',
    'fjob': 'services',
    'reason': 'course',
    'guardian': 'mother'
}

# Send POST request
response = requests.post(url, json=data)

# Print the response
print(response.text)