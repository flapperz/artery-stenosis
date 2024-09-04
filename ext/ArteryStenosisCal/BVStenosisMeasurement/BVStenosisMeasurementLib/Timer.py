import time


class Timer:
    def __init__(self, isDebug=False):
        self.isDebug = isDebug
        self.startDict = dict()
        self.stopDict = dict()
        self.currentEntry = None
        self.timerBorn = time.time()

    def start(self, entry):
        if entry in self.startDict:
            raise Exception(f'entry {entry} already exist')
        self.startDict[entry] = time.time()
        self.currentEntry = entry

    def stop(self):
        if self.currentEntry:
            self.stopDict[self.currentEntry] = time.time()
            currentEntry = self.currentEntry
            self.currentEntry = None

            if self.isDebug:
                elapsedTime = self.stopDict[currentEntry] - self.startDict[currentEntry]
                print(
                    f'Timer Report: {currentEntry} take {elapsedTime:.3f} s.'
                )

    def generateReport(self):
        timerDie = time.time()

        report = ""
        report += "Timer Report \n"
        report += "------------ \n"
        report += f"Overall Time: {timerDie - self.timerBorn:.3f} \n"
        for k in self.startDict.keys():
            if k in self.stopDict:
                elapsedTime = self.stopDict[k] - self.startDict[k]
                report += f'{k}: {elapsedTime:.3f} s.\n'
        return report
