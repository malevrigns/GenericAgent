import sys,time
sys.path.insert(0,'C:/Users/ASUS/Documents/genericagent')
from TMWebDriver import TMWebDriver
m=TMWebDriver()
print('tmwd master started',flush=True)
while True:
    time.sleep(3600)
