from ..IPU.ipu import IPU
from .rf import RegisterFile
from .controller import TileController
from .act import Activation
from .adder_tree import AdderTree
from .router import Router
from C_Graph.variable import Variable
from .buffer import Buffer
from functools import reduce


class Tile:

    def __init__(self, id, IPU_num=4):
        self.id = id
        self.rf = RegisterFile(data_depth=64, index_num=256, data_num=1024)
        self.buffer = Buffer(data_depth=64)  # 本地缓存1MB, 20位二进制地址 = 5位十六进制地址
        self.bf_ptr = 0  # 本地缓存指针，按字节编址，5位十六进制
        self.IPU_list = []
        self.op_code = []
        self.tileController = TileController()
        self.act = Activation()
        self.adderTree = AdderTree()
        self.router = Router()
        self.input_indexReg = {}  # 层数和索引寄存器地址列表的字典，所有该层的输入都会经过这些索引寄存器
        self.mov_reg = {}  # 层数和移动暂存寄存器地址列表的字典，所有该层的中间输出都会经过这些数据寄存器
        self.output_indexReg = {}  # 层数和索引寄存器地址列表的字典，所有该层的输出都会经过这些索引寄存器
        self.layer_i_ptr = {}  # 层i输入数据首地址表
        self.row_indexReg = {}  # 键：IPU_id, 值：该IPU内XB行的索引寄存器地址列表（二进制）
        self.IPU_usage = []  # IPU使用表，IPU_usage[i]=1 代表ipu_id=i的IPU被使用了
        self.used = False  # 当且仅当 IPU_usage[i]=1 for i=0~n-1 时used=True
        for i in range(IPU_num):
            if i < IPU_num // 2:
                ipu = IPU(id=i, type="X")
            else:
                ipu = IPU(id=i, type="W")
            self.IPU_list.append(ipu)
        self.IPU_usage = [0] * IPU_num

    def initInputIndexRegs(self, layer_i, vector_size):
        index_reg_addrs = self.rf.initDataIndexRegs(vector_size)
        self.input_indexReg[layer_i] = index_reg_addrs

    def initMovRegs(self, layer_i, vector_size):
        if layer_i not in self.mov_reg:
            # 可能会出现同一个tile中存放了相邻的两行的分块权重，则可以共用一组mov_regs
            mov_addrs = self.rf.initDataRegs(vector_size)
            self.mov_reg[layer_i] = mov_addrs

    def initOutputIndexRegs(self, layer_i, vector_size):
        index_reg_addrs = self.rf.initDataIndexRegs(vector_size)
        self.output_indexReg[layer_i] = index_reg_addrs

    def deployWeight(self, IPU_id, weight: Variable):
        self.IPU_list[IPU_id].deployWeight(weight)

    def initRowIndexRegs(self, IPU_id, act_rows):
        self.row_indexReg[IPU_id] = self.rf.initRowIndexRegs(act_rows)

    def initXBar(self, IPU_id, W_height, W_width):
        return self.IPU_list[IPU_id].initXBar(W_height, W_width)

    def initXBarRep(self, IPU_id, W_height, W_width, replica_num):
        return self.IPU_list[IPU_id].initXBarRep(W_height, W_width, replica_num)

    def initXBarRepOverlap(self, IPU_id, W_height, W_width, replica_num, offset):
        return self.IPU_list[IPU_id].initXBarRepOverlap(W_height, W_width, replica_num, offset)

    def load(self, dst_addr, vec_data):
        self.rf.load(dst_addr, vec_data)

    def store(self, src_addr, vec_width):
        return self.rf.store(src_addr, vec_width)

    def dot(self, IPU_id, src1, src2, dst):
        # 生成以src1和src2为起始地址的索引寄存器中存的掩码
        row_mask = self.rf.getMask(src1, 4)
        data_mask = list(self.rf.getMask(src2, 16))
        # print(f"data_mask = {data_mask}")
        dst_mask = list(self.rf.getMask(dst, 16))
        # 取出data mask中的data regs的值，形成V向量
        reg_index = [i for i in range(len(data_mask)) if data_mask[i] == '1']
        # print("reg_index: ", reg_index)
        data_list = self.rf.getListOfValue(reg_index, group="data")
        # print("in tile: ", data_list)
        I_out = self.IPU_list[IPU_id].dot(row_mask, V=data_list)
        dst_addr = [i for i in range(len(dst_mask)) if dst_mask[i] == '1']
        self.rf.setListOfValue(dst_addr, I_out, group='data')
        return I_out

    # 根据本层输入信息，为(im2col后的)输入数据预分配buffer空间
    def initBuffer(self, layer_i, batch_size, layer_params, layer_info):

        if layer_params["op_code"] == "conv":
            out_channel, in_channel, K_h, K_w = layer_params["W_shape"]
            in_c, h, w = layer_params["input_shape"]
            stride, padding = layer_params["stride"], layer_params["padding"]

            if isinstance(stride, tuple):
                stride_h, stride_w = stride
            else:
                stride_h = stride_w = stride

            if isinstance(padding, tuple):
                padding_h, padding_w = padding
            else:
                padding_h = padding_w = padding

            y_h = (h - K_h + 2 * padding_h) // stride_h + 1
            y_w = (w - K_w + 2 * padding_w) // stride_w + 1

            if "act_rows" in layer_info and "act_cols" in layer_info:
                replica_num = layer_info.get("replica_num", None)
                if replica_num is None:
                    raise Exception("Replica_num does not exist!")
                if replica_num > 1:
                    in_shape = [batch_size, y_h * y_w // replica_num, in_channel * K_h * K_w * replica_num]
                else:
                    in_shape = [batch_size, y_h * y_w, in_channel * K_h * K_w]
            else:
                # TODO: 中大规模的im2col
                in_shape = None


        else:
           in_shape = [batch_size, layer_params['input_shape']]


        self.layer_i_ptr[layer_i] = self.bf_ptr
        offset = reduce(lambda x,y: x*y, in_shape)
        # buffer指针bf_ptr预分配
        self.bf_ptr += offset * (self.buffer.data_depth // 8)

        return in_shape

    def assignX2Buffer(self, convertedX: Variable, layer_i):
        start_addr = self.layer_i_ptr[layer_i]
        c_X = convertedX.data.reshape(-1)
        for i in range(len(c_X)):
            self.buffer.write(start_addr + i * (self.buffer.data_depth // 8), c_X[i])
