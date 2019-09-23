import json


def config(path: str):
    config = json.loads(open(path, 'r').read())
    return config
