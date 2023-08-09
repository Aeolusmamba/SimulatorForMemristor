class Operation:
    pass


class AddOperation(Operation):
    src1_reg = None
    src2_reg = None
    dst_reg = None

    def setSrc1Reg(self, reg):
        self.src1_reg = reg

    def setSrc2Reg(self, reg):
        self.src2_reg = reg

    def setDstReg(self, reg):
        self.dst_reg = reg

    def getSrc1Reg(self):
        return self.src1_reg

    def getSrc2Reg(self):
        return self.src2_reg

    def getDstReg(self):
        return self.dst_reg


class SubOperation(Operation):
    src1_reg = None
    src2_reg = None
    dst_reg = None

    def setSrc1Reg(self, reg):
        self.src1_reg = reg

    def setSrc2Reg(self, reg):
        self.src2_reg = reg

    def setDstReg(self, reg):
        self.dst_reg = reg

    def getSrc1Reg(self):
        return self.src1_reg

    def getSrc2Reg(self):
        return self.src2_reg

    def getDstReg(self):
        return self.dst_reg


class LoadOperation(Operation):

    def __init__(self, rs_addr, rd_addr):
        self.rs_addr, self.rd_addr = rs_addr, rd_addr




class DotOperation(Operation):
    src1_reg = None
    src2_reg = None
    dst_reg = None

    def setSrc1Reg(self, reg):
        self.src1_reg = reg

    def setSrc2Reg(self, reg):
        self.src2_reg = reg

    def setDstReg(self, reg):
        self.dst_reg = reg

    def getSrc1Reg(self):
        return self.src1_reg

    def getSrc2Reg(self):
        return self.src2_reg

    def getDstReg(self):
        return self.dst_reg


class MulOperation(Operation):
    src1_reg = None
    src2_reg = None
    dst_reg = None

    def setSrc1Reg(self, reg):
        self.src1_reg = reg

    def setSrc2Reg(self, reg):
        self.src2_reg = reg

    def setDstReg(self, reg):
        self.dst_reg = reg

    def getSrc1Reg(self):
        return self.src1_reg

    def getSrc2Reg(self):
        return self.src2_reg

    def getDstReg(self):
        return self.dst_reg


class MaskOperation(Operation):

    src_reg = None
    dst_reg = None
    imm = None

    def setSrcReg(self, reg):
        self.src_reg = reg

    def setDstReg(self, reg):
        self.dst_reg = reg

    def setImm(self, imm):
        self.imm = imm

    def getSrcReg(self):
        return self.src_reg

    def getDstReg(self):
        return self.dst_reg

    def getImm(self):
        return self.imm


class MovOperation(Operation):
    src_reg = None
    dst_reg = None

    def setSrcReg(self, reg):
        self.src_reg = reg

    def setDstReg(self, reg):
        self.dst_reg = reg

    def getSrcReg(self):
        return self.src_reg

    def getDstReg(self):
        return self.dst_reg


class LUTOperation(Operation):
    src_reg = None
    dst_reg = None

    def setSrcReg(self, reg):
        self.src_reg = reg

    def setDstReg(self, reg):
        self.dst_reg = reg

    def getSrcReg(self):
        return self.src_reg

    def getDstReg(self):
        return self.dst_reg


class LoadOperation(Operation):
    src_reg = None
    memAddr = None

    def setSrcReg(self, reg):
        self.src_reg = reg

    def setMemAddr(self, memAddr):
        self.memAddr = memAddr

    def getSrcReg(self):
        return self.src_reg

    def getMemAddr(self):
        return self.memAddr


class StoreOperation(Operation):
    src_reg = None
    memAddr = None

    def setSrcReg(self, reg):
        self.src_reg = reg

    def setMemAddr(self, memAddr):
        self.memAddr = memAddr

    def getSrcReg(self):
        return self.src_reg

    def getMemAddr(self):
        return self.memAddr


class TileOperation(Operation):
    pass

class UpdateOperation(Operation):
    def update(self):
        pass
