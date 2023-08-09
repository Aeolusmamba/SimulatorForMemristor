


class InstructionMemory():

    memory = {}

    def __init__(self):
        pass

    def read(self, addr):
        if addr in self.memory:
            return self.memory[addr]
        else:
            return None

    def write(self, addr, data):
        self.memory[addr] = data
