class PC:

    def __init__(self, inst_depth):
        self.program_counter = 0  # PC的值，即指令(字节)的地址
        self.pc_offset = inst_depth // 8  # pc 的偏移量 == 指令字地址长度

    def step(self):
        self.program_counter += self.pc_offset

    def jump(self, offset):
        self.program_counter += offset*self.pc_offset

    def set(self, value):
        self.program_counter = value

    def reset(self, istart_addr):
        self.program_counter = istart_addr
