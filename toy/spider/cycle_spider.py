import time
import os
import threading

def createTimer():
    t = threading.Timer(10, repeat)
    t.start()

def repeat():
    cmd = 'scrapy crawl hermers'
    os.system(cmd)
    createTimer()

if __name__ == '__main__':
    createTimer()