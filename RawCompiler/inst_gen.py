from operations import *


class InstructionGenerator():
    def __init__(self):
        pass

    def instGen(self, operation):
        inst = ""
        if isinstance(operation, AddOperation):
            inst += "add "
            inst += operation.getSrc1Reg() + " "
            inst += operation.getSrc2Reg() + " "
            inst += operation.getDstReg()
        elif isinstance(operation, DotOperation):
            inst += "dot "
            inst += operation.getSrc1Reg() + " "
            inst += operation.getSrc2Reg() + " "
            inst += operation.getDstReg()
        elif isinstance(operation, UpdateOperation):
            inst += "update"
        elif isinstance(operation, MulOperation):
            inst += "mul "
            inst += operation.getSrc1Reg() + " "
            inst += operation.getSrc2Reg() + " "
            inst += operation.getDstReg()
        elif isinstance(operation, SubOperation):
            inst += "sub "
            inst += operation.getSrc1Reg() + " "
            inst += operation.getSrc2Reg() + " "
            inst += operation.getDstReg()
        elif isinstance(operation, MaskOperation):
            inst += "mask "
            inst += operation.getSrcReg() + " "
            inst += operation.getDstReg() + " "
            inst += operation.getImm()
        elif isinstance(operation, MovOperation):
            inst += "mov "
            inst += operation.getSrcReg() + " "
            inst += operation.getDstReg()
        elif isinstance(operation, LUTOperation):
            inst += "lut "
            inst += operation.getSrcReg() + " "
            inst += operation.getDstReg()
        elif isinstance(operation, LoadOperation):
            inst += "lw "
            inst += operation.getSrcReg() + " "
            inst += operation.getMemAddr()
        elif isinstance(operation, StoreOperation):
            inst += "sw "
            inst += operation.getSrcReg() + " "
            inst += operation.getMemAddr()
        return inst
