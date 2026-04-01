import json

with open("slots.json") as f:
    data = json.load(f)

print(data)