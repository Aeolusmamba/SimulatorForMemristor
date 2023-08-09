from .data_mem import DataMemory
from C_Graph.variable import Variable
from C_Graph.operator import Operator
from .inst_mem import InstructionMemory
from .pc import PC


class Controller():

    def __init__(self):
        self.pc = PC(inst_depth=64)

    def execute_inst(self, instMemory):
        pass

