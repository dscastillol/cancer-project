import requests
import json
from pprint import pprint


with open(r'images\ISIC_0000001') as f:
    data = json.load(f)

pprint(data)