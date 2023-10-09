# Local Buffer

class Buffer:

    data_depth = 64  # 数据字长
    buffer = {}

    def __init__(self, data_depth=64):
        self.data_depth = data_depth

    def read(self, addr):  # read a word
        if addr in self.buffer:  # 地址有效
            return self.buffer[addr]
        else:
            return None

    def write(self, addr, data):  # write a word
        self.buffer[addr] = data
