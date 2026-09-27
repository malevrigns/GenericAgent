import sys,os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from TMWebDriver import TMWebDriver
d=TMWebDriver(host='127.0.0.1',port=18765)
import time
while True: time.sleep(3600)
