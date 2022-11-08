import time
class MaintletTimer:

    def __init__(self, measuredTask, fileSize = 0):
        self.startTime = 0
        self.measuredTask = measuredTask
        self.fileSize = fileSize
    
    def calculateBandwidth(self, time, size):
        """
        Calculate the bandwidth
        Args:
            time (float): unit: s 
            size (int): unit: byte 
        Returns:
            float: MB/s
        """        
        return round((size/1024/1024) / (time), 3) 
        

    def __enter__(self):
        self.startTime = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapsedTimeInSecond = time.time() - self.startTime
        if self.fileSize == 0:
            # no need to calculate the bandwidth
            print(f"{self.measuredTask} takes {round(elapsedTimeInSecond*1000, 2)} ms")
        else:
            print(f"{self.measuredTask} takes {round((time.time() - self.startTime)*1000, 2)} ms. Averaged Bandwidth {self.calculateBandwidth(elapsedTimeInSecond, self.fileSize)} MB/s") 

        return True