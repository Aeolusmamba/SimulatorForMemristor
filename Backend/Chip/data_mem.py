from C_Graph.variable import Variable, GLOBAL_VARIABLE_SCOPE


class MemInput:

    id = None
    input_variable = None  # 现在暂时只放一个样本数据
    input_address = None  # with the form of Tile_id_IPU_id
    back_address = None  # with the form of Tile_id_IPU_id


    def __init__(self, input_variable):
        self.input_variable = input_variable

    def setId(self, id):
        self.id = id

    def setInputAddr(self, addr):
        self.input_address = addr

    def setBackAddr(self, addr):
        self.back_address = addr


class DataMemory:

    data_depth = 64  # 数据字长
    memory = {}

    def __init__(self, data_depth=64):
        self.data_depth = data_depth

    def read(self, addr):  # read a word
        if addr in self.memory:  # 地址有效
            return self.memory[addr]
        else:
            return None

    def write(self, addr, data):  # write a word
        self.memory[addr] = data
