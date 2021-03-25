import time
import datetime
from dateutil.relativedelta import relativedelta

class Timer(object):
    """Computes elapsed time."""
    def __init__(self, name):
        self.name = name
        self.running = True
        self.total = 0
        self.start = round(time.time(), 2)
        self.intervalTime = round(time.time(), 2)
        self.start_time = datetime.datetime.now()
        print("<> <> <> Starting Timer [{}] <> <> <>".format(self.name))

    def reset(self):
        self.running = True
        self.total = 0
        self.start = round(time.time(), 2)
        return self

    def interval(self, intervalName=''):
        intervalTime = self._to_hms(round(time.time() - self.intervalTime, 2))
        print("<> <> Timer [{}] <> <> Interval [{}]: {} <> <>".format(self.name, intervalName, intervalTime))
        self.intervalTime = round(time.time(), 2)
        return intervalTime

    def stop(self):
        if self.running:
            self.running = False
            self.total += round(time.time() - self.start, 2)
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = round(time.time(), 2)
        return self

    def time(self):
        if self.running:
            return round(self.total + time.time() - self.start, 2)
        return self.total

    def finish(self):
        if self.running:
            self.running = False
            self.total += round(time.time() - self.start, 2)
            elapsed = self._to_hms(self.total)
        print("<> <> <> Finished Timer [{}] <> <> <> Total time elapsed: {} <> <> <>".format(self.name, elapsed))

    def _to_hms(self, seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return "%dh %02dm %02ds" % (h, m, s)

    def remains(self, total_task_num,done_task_num):
        now  = datetime.datetime.now()
        #print(now-start)  # elapsed time
        left = (total_task_num - done_task_num) * (now - self.start_time) / done_task_num
        sec = int(left.total_seconds())
        
        rt = relativedelta(seconds=sec)
     
        return "{:02d} hours {:02d} minutes {:02d} seconds".format(int(rt.hours), int(rt.minutes), int(rt.seconds))