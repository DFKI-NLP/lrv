import datetime
import sys


def log(msg):
    now = datetime.datetime.now()
    print(str(now) + ' ' + msg)
    sys.stdout.flush()
