from ..IPU.ipu import IPU
from .rf import RegisterFile
from .controller import TileController
from .act import Activation
from .adder_tree import AdderTree
from .router import Router
from C_Graph.variable import Variable


class Tile:

    def __init__(self, id, IPU_num=4):
        self.id = id
        self.rf = RegisterFile(data_depth=64, index_num=256, data_num=512)
        self.IPU_list = []
        self.op_code = []
        self.tileController = TileController()
        self.act = Activation()
        self.adderTree = AdderTree()
        self.router = Router()
        self.input_indexReg = {}  # 样本序号和索引寄存器地址列表的字典，用于记录该样本将被分配到哪些输入索引寄存器中去
        self.output_indexReg = {}  # 样本序号和索引寄存器地址列表的字典，用于记录该样本将被分配到哪些输出索引寄存器中去
        self.row_indexReg = None  # XB行的索引寄存器地址列表，寄存器存mask，记录哪些行将参与计算
        self.IPU_usage = []  # IPU使用表，IPU_usage[i]=1 代表ipu_id=i的IPU被使用了
        self.used = False  # 当且仅当 IPU_usage[i]=1 for i=0~n-1 时used=True
        for i in range(IPU_num):
            if i < IPU_num // 2:
                ipu = IPU(id=i, type="X")
            else:
                ipu = IPU(id=i, type="W")
            self.IPU_list.append(ipu)
        self.IPU_usage = [0] * IPU_num

    def initInputIndexRegs(self, input_i, vector_size):
        index_reg_addrs = self.rf.initDataIndexRegs(vector_size)
        self.input_indexReg[input_i] = index_reg_addrs

    def initOutputIndexRegs(self, input_i, vector_size):
        index_reg_addrs = self.rf.initDataIndexRegs(vector_size)
        self.output_indexReg[input_i] = index_reg_addrs

    def deployWeight(self, IPU_id, weight: Variable):
        self.IPU_list[IPU_id].deployWeight(weight)
        # 返回部署有权重的行的列表，如：W 9*128 -> [0~8]
        act_rows = list(range(weight.shape[0]))
        # print(f"act_rows={act_rows}")
        return act_rows

    def initRowIndexRegs(self, act_rows):
        self.row_indexReg = self.rf.initRowIndexRegs(act_rows)

    def load(self, dst_addr, vec_data):
        self.rf.load(dst_addr, vec_data)

    def store(self, src_addr, vec_width):
        return self.rf.store(src_addr, vec_width)

    def dot(self, IPU_id, src1, src2, dst):
        # 生成以src1和src2为起始地址的索引寄存器中存的掩码
        row_mask = self.rf.getMask(src1)
        data_mask = list(self.rf.getMask(src2))
        dst_mask = list(self.rf.getMask(dst))
        # 取出data mask中的data regs的值，形成V向量
        reg_index = [i for i in range(len(data_mask)) if data_mask[i] == '1']
        # print("reg_index: ", reg_index)
        data_list = self.rf.getListOfValue(reg_index, group="data")
        # print("in tile: ", data_list)
        I_out = self.IPU_list[IPU_id].dot(row_mask, V=data_list)
        dst_addr = [i for i in range(len(dst_mask)) if dst_mask[i] == '1']
        self.rf.setListOfValue(dst_addr, I_out, group='data')
        return I_out
