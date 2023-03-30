import sys


class Logger(object):
    def __init__(self, filename: str="log.txt") -> None:
        self.terminal = sys.stdout
        sys.stdout = self
        self.log = open(filename, "a")

    def close(self) -> None:
        sys.stdout = self.terminal
        self.log.close()

    def write(self, message) -> None:
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()
