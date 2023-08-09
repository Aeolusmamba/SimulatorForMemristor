class Reg:

    id = 0
    used = False

    def __init__(self, id=0, value=0):
        self.id = id
        self.value = value

    def read(self):
        return self.value

    def write(self, val):
        self.value = val
