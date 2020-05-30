import time
import datetime

from config import PROGRESS_BAR

class ProgressBar:

    def __init__(self, length):
        self.length = length
        self.i = 0
        self.show()
        self.init_time = time.time()
        self.done = False

    def show(self):
        progress = int(self.i / self.length * PROGRESS_BAR['width'])
        print('[' + PROGRESS_BAR['positive'] * progress + PROGRESS_BAR['negative'] * (PROGRESS_BAR['width'] - progress) + '] ' + str(self.i) + ' / ' + str(self.length) + ' ' + str(datetime.timedelta(seconds=int(time.time() - self.init_time))), end='\r')

    def increment(self):
        if self.done: return
        self.i += 1
        self.show()
        if self.i >= self.length:
            self.done = True
            print("Finished in", str(datetime.timedelta(seconds=int(time.time() - self.init_time))))