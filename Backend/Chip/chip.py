from ..Tile.tile import Tile
import numpy as np
from .inst_mem import InstructionMemory
from .inst_decoder import InstructionDecoder
from .data_mem import DataMemory
from .accumulator import Accumulator
from C_Graph.variable import Variable
from .pc import PC
import math
from functools import reduce


class Chip:

    def __init__(self, id, tile_num=256):
        self.id = id
        self.base_size = 256
        self.tile_list = []
        self.instMemory = InstructionMemory()  # 4GB，32位地址
        self.inst_depth = 64  # 64位指令字长
        self.instDecoder = InstructionDecoder()
        self.pc = PC(inst_depth=64)
        self.dataMemory = DataMemory()  # 4GB，32位二进制地址 = 8位十六进制地址
        self.accumulator = Accumulator()
        self.dmem_ptr = 0  # 数据内存指针，按字节编址，32位地址，十六进制字符串形式存储（8位）
        self.imem_ptr = 0  # 指令内存指针，按字节编址，32位地址，十六进制字符串形式存储（8位）
        self.tile_usage = []  # tile使用表，tile_usage[i]=1 代表tile_id=i的tile被使用了
        self.used = False  # 当且仅当 tile_usage[i]=1 for i=0~n-1 时used=True
        self.dmem_input_table = {}  # data memory的”输入层“数据地址索引表，键：sample的唯一标识（暂时为batch_i），值：该sample的起始地址
        self.dmem_output_table = {}  # data memory的”输出层“数据地址索引表，键：sample的唯一标识（暂时为batch_i），值：该sample的起始地址
        self.imem_table = {}  # key: insts的PC word, value: inst mem中的（起始）地址
        for i in range(tile_num):
            tile = Tile(id=i)
            self.tile_list.append(tile)
        self.tile_usage = [0] * tile_num

    def assignInput2Mem(self, first_input: Variable, layer_i, layer_params: dict, layer_info: dict, first_layer):
        batch_size = first_input.shape[0]
        addrs = layer_info["addrs"]
        converted_Xshape = None
        for addr in addrs:
            tile_id, IPU_id = addr.split("_")
            tile_id, IPU_id = int(tile_id), int(IPU_id)

            if first_layer is not True:
                # 第一层以后的中间层对上一层的输出数据在本层的local buffer中进行初始化
                # 本层读取的输入即来自本层的local buffer

                converted_Xshape = self.initBuffer(tile_id, layer_i, batch_size, layer_params, layer_info)
            else:
                converted_Xshape = first_input.shape

        # 提前将输入层写入到内存中，总体目标是仅输入输出会访存，中间的数据流都在本地缓存中
        if first_layer:
            for i in range(batch_size):
                self.dmem_input_table[i] = self.dmem_ptr
                sample = first_input.data[i]  # [784, 9]
                # 每个样本连续存放在数据内存中，样本与样本之间可以不连续
                # print("storing input data...")
                # print(f"initial storing addr: ", self.dmem_ptr)
                for data in np.reshape(sample, -1):
                    # print(self.dmem_ptr)
                    self.dataMemory.write(self.dmem_ptr, data)  # write a word
                    self.dmem_ptr += self.dataMemory.data_depth // 8  # 加上一个字的地址偏移，按字节编址

        return converted_Xshape

    def assignOutput2Mem(self, output_shape: list, layer_info: dict):
        # 根据内存指针dmem_ptr开始预分配
        batch_size = output_shape[0]
        sample_addr_offset = reduce(lambda x, y: x*y, output_shape[1:]) * self.dataMemory.data_depth // 8
        for i in range(batch_size):
            self.dmem_output_table[i] = self.dmem_ptr
            self.dmem_ptr += sample_addr_offset

    def assignX2Mem(self, result_X: Variable):
        # 实际写入最终的输出结果
        batch_size = result_X.shape[0]
        for i in range(batch_size):
            ptr = self.dmem_output_table[i]
            for x in result_X.data[i].reshape(-1):
                self.dataMemory.write(ptr, x)
                ptr += self.dataMemory.data_depth // 8

    def initSScale(self, layer_i, W_shape, type="conv"):
        addrs = []
        layer_info = {}
        addr = ""
        act_rows = act_cols = 0
        if type == "conv":
            out_channel, in_channel, K_h, K_w, S_x, S_y = W_shape
            W_height = in_channel * K_h * K_w
            W_width = out_channel
            # if in_channel == 1:
            #     # 单通道，对角线重叠复制
            #     offset = K_h * S_y
            #     col_num = self.base_size // W_width
            #     row_num = (self.base_size - W_height) // offset + 1
            #     replica_num = min(col_num, row_num)
            #     layer_info["replica_num"] = replica_num
            #     tile_id, IPU_id = self.initSingleIPU()
            #     act_rows, act_cols = self.initXBarRepOverlap(tile_id, IPU_id, W_height, W_width, replica_num, offset)
            #     addr = str(tile_id) + "_" + str(IPU_id)
            # else:
            #     # 多通道，非重叠复制

            col_num = self.base_size // W_width
            row_num = self.base_size // W_height
            replica_num = min(col_num, row_num)
            layer_info["replica_num"] = replica_num
            tile_id, IPU_id = self.initSingleIPU()
            act_rows, act_cols = self.initXBarRep(tile_id, IPU_id, W_height, W_width, replica_num)
            addr = str(tile_id) + "_" + str(IPU_id)

        else:
            W_height, W_width = W_shape
            tile_id, IPU_id = self.initSingleIPU()
            act_rows, act_cols = self.initXBar(tile_id, IPU_id, W_height, W_width)
            addr = str(tile_id) + "_" + str(IPU_id)

        print(f"Initilizing I\O registers of layer_i {layer_i} in tile {tile_id}, IPU {IPU_id}...")
        self.tile_list[tile_id].initInputIndexRegs(layer_i, act_rows)
        print(f"{act_rows} input index registers are ready!")
        self.tile_list[tile_id].initOutputIndexRegs(layer_i, act_cols)
        print(f"{act_cols} output index registers are ready!")
        self.tile_list[tile_id].initRowIndexRegs(IPU_id, list(range(act_rows)))

        addrs.append(addr)
        layer_info["type"] = type
        layer_info["addrs"] = addrs
        layer_info["act_rows"] = act_rows
        layer_info["act_cols"] = act_cols
        return layer_info

    def initMScale(self, layer_i, W_shape, type="conv"):
        layer_info = {}
        addrs = []
        XBar_hshape = XBar_wshape = []
        if type == "conv":
            out_channel, in_channel, K_h, K_w, S_x, S_y = W_shape
            W_height = in_channel * K_h * K_w
            W_width = out_channel
            W_IPU_shape = [W_height, W_width]
            # 对于中规模和大规模的NN，由于暂时是每个IPU只放一个子矩阵，
            # 故XBar_hshape和XBar_wshape可以替代小规模情况中的act_rows和act_cols
            addrs, XBar_hshape, XBar_wshape = self.initMultipleIPU(W_IPU_shape)
        else:
            addrs, XBar_hshape, XBar_wshape = self.initMultipleIPU(W_shape)

        for a_i in range(len(addrs)):
            tile_id, IPU_id = addrs[a_i].split('_')
            tile_id, IPU_id = int(tile_id), int(IPU_id)
            self.tile_list[tile_id].initInputIndexRegs(layer_i, XBar_hshape[a_i // len(XBar_wshape)])
            self.tile_list[tile_id].initOutputIndexRegs(layer_i, XBar_wshape[a_i % len(XBar_wshape)])
            self.tile_list[tile_id].initRowIndexRegs(IPU_id, list(range(XBar_hshape[a_i // len(XBar_wshape)])))

            if len(XBar_hshape) > 1:
                # 有纵向分块权重，则下一行所在tile的rf（也有可能是一个tile）接收上一行的输出
                current_loc = a_i  # 当前向量所处的块编号（从0开始，行优先排列）
                current_hloc, current_wloc = current_loc // len(XBar_wshape), current_loc % len(XBar_wshape)
                if current_hloc > 0:
                    self.initMovRegs(tile_id, layer_i, XBar_wshape[current_wloc])

        layer_info["type"] = type
        layer_info["addrs"] = addrs
        layer_info["XBar_hshape"] = XBar_hshape
        layer_info["XBar_wshape"] = XBar_wshape
        return layer_info

    def initLScale(self, layer_i, W_shape, type="conv"):
        layer_info = {}
        addrs = []
        XBar_hshape = XBar_wshape = []
        if type == "conv":
            out_channel, in_channel, K_h, K_w, S_x, S_y = W_shape
            W_height = in_channel * K_h * K_w
            W_width = out_channel
            W_IPU_shape = [W_height, W_width]
            addrs, XBar_hshape, XBar_wshape = self.initMultipleIPU(W_IPU_shape)
        else:
            addrs, XBar_hshape, XBar_wshape = self.initMultipleIPU(W_shape)

        for a_i in range(len(addrs)):
            tile_id, IPU_id = addrs[a_i].split('_')
            tile_id, IPU_id = int(tile_id), int(IPU_id)
            self.tile_list[tile_id].initInputIndexRegs(layer_i, XBar_hshape[a_i // len(XBar_wshape)])
            self.tile_list[tile_id].initOutputIndexRegs(layer_i, XBar_wshape[a_i % len(XBar_wshape)])
            self.tile_list[tile_id].initRowIndexRegs(IPU_id, list(range(XBar_hshape[a_i // len(XBar_wshape)])))

            if len(XBar_hshape) > 1:
                # 有纵向分块权重，则下一行所在tile的rf（也有可能是一个tile）接收上一行的输出
                current_loc = a_i  # 当前向量所处的块编号（从0开始，行优先排列）
                current_hloc, current_wloc = current_loc // len(XBar_wshape), current_loc % len(XBar_wshape)
                if current_hloc > 0:
                    self.initMovRegs(tile_id, layer_i, XBar_wshape[current_wloc])

        layer_info["type"] = type
        layer_info["addrs"] = addrs
        layer_info["XBar_hshape"] = XBar_hshape
        layer_info["XBar_wshape"] = XBar_wshape
        return layer_info

    def initSingleIPU(self):
        tile_num = len(self.tile_list)
        tile_cnt = 0
        IPU_cnt = 0

        while tile_cnt < tile_num:  # check usage
            if self.tile_usage[tile_cnt] == 0:
                if self.tile_list[tile_cnt].IPU_usage[2] == 0:
                    self.tile_list[tile_cnt].IPU_usage[2] = 1
                    IPU_cnt = 2
                    break
                elif self.tile_list[tile_cnt].IPU_usage[3] == 0:
                    self.tile_list[tile_cnt].IPU_usage[3] = 1
                    IPU_cnt = 3
                    break
                else:
                    self.tile_usage[tile_cnt] = 1
            tile_cnt = tile_cnt + 1
        if tile_cnt == tile_num:
            raise Exception("Running out of Tile!")
        return tile_cnt, IPU_cnt

    def initMultipleIPU(self, W_shape):
        W_height, W_width = W_shape
        tile_num = len(self.tile_list)
        tile_cnt = 0
        addrs = []
        row_XBar_num = math.ceil(W_height / self.base_size)  # 均匀分块，平摊计算量
        W_XBar_h = W_height // row_XBar_num  # W在XBar中的高度
        XBar_hshape = []
        for i in range(row_XBar_num - 1):
            XBar_hshape.append(W_XBar_h)
        XBar_hshape.append(W_height - (row_XBar_num - 1) * W_XBar_h)  # 多出的零头放在最后一个里面
        col_XBar_num = math.ceil(W_width / self.base_size)
        W_XBar_w = W_width // col_XBar_num  # W在XBar中的宽度
        XBar_wshape = []
        for i in range(col_XBar_num - 1):
            XBar_wshape.append(W_XBar_w)
        XBar_wshape.append(W_width - (col_XBar_num - 1) * W_XBar_w)  # 多出的零头放在最后一个里面

        for i in range(len(XBar_hshape)):
            for j in range(len(XBar_wshape)):
                while tile_cnt < tile_num:  # check usage
                    if self.tile_usage[tile_cnt] == 0:
                        if self.tile_list[tile_cnt].IPU_usage[2] == 0:
                            self.tile_list[tile_cnt].IPU_usage[2] = 1
                            IPU_cnt = 2
                            addr = str(tile_cnt) + "_" + str(IPU_cnt)
                            self.initXBar(tile_cnt, IPU_cnt, XBar_hshape[i], XBar_wshape[j])
                            addrs.append(addr)
                            break
                        elif self.tile_list[tile_cnt].IPU_usage[3] == 0:
                            self.tile_list[tile_cnt].IPU_usage[3] = 1
                            IPU_cnt = 3
                            addr = str(tile_cnt) + "_" + str(IPU_cnt)
                            self.initXBar(tile_cnt, IPU_cnt, XBar_hshape[i], XBar_wshape[j])
                            addrs.append(addr)
                            break
                        else:
                            self.tile_usage[tile_cnt] = 1
                    tile_cnt = tile_cnt + 1
                if tile_cnt == tile_num:
                    raise Exception("Running out of Tile!")
        return addrs, XBar_hshape, XBar_wshape

    def initXBar(self, tile_id, IPU_id, W_height, W_width):
        return self.tile_list[tile_id].initXBar(IPU_id, W_height, W_width)

    def initXBarRep(self, tile_id, IPU_id, W_height, W_width, replica_num):
        return self.tile_list[tile_id].initXBarRep(IPU_id, W_height, W_width, replica_num)

    def initXBarRepOverlap(self, tile_id, IPU_id, W_height, W_width, replica_num, offset):
        return self.tile_list[tile_id].initXBarRepOverlap(IPU_id, W_height, W_width, replica_num, offset)

    def initInputIndexRegs(self, tile_id, layer_i, vector_size):
        self.tile_list[tile_id].initInputIndexRegs(layer_i, vector_size)

    def initMovRegs(self, tile_id, layer_i, vector_size):
        self.tile_list[tile_id].initMovRegs(layer_i, vector_size)

    def initOutputIndexRegs(self, tile_id, layer_i, vector_size):
        self.tile_list[tile_id].initOutputIndexRegs(layer_i, vector_size)

    def initBuffer(self, tile_id, layer_i, batch_size, layer_params, layer_info):
        return self.tile_list[tile_id].initBuffer(layer_i, batch_size, layer_params, layer_info)

    def deployWeight(self, tile_id, IPU_id, weight: Variable):
        self.tile_list[tile_id].deployWeight(IPU_id, weight)

    def storeInstructions(self, inst_list):
        inst_word_offset = self.inst_depth // 8  # 指令字地址偏移，64/8=8
        istart_addr = self.imem_ptr
        for i in range(len(inst_list)):
            self.imem_table[i] = self.imem_ptr
            self.instMemory.write(self.imem_ptr, inst_list[i])
            self.imem_ptr += inst_word_offset
        iend_addr = self.imem_ptr
        return istart_addr, iend_addr

    def execute(self, inst_list, addrs):
        result = []
        dmem_word_len = self.dataMemory.data_depth // 8
        self.load_data = []

        for inst_i in range(len(inst_list)):
            addr = addrs[inst_i % len(addrs)]
            inst = inst_list[inst_i]
            tile_id, IPU_id = addr.split("_")
            tile_id, IPU_id = int(tile_id), int(IPU_id)

            op_code, operands = self.instDecoder.decode(inst)

            if op_code == "load":

                mem_addr = int(operands[1], 16)  # 在数据内存中的地址，16进制转成10进制
                vec_width = int(operands[2], 16)

                vec_data = []
                for i in range(vec_width):
                    if len(operands[1]) == 8:
                        # 地址指向数据内存
                        vec_data.append(self.dataMemory.read(mem_addr + i * dmem_word_len))
                    else:
                        # 地址指向缓存
                        vec_data.append(self.tile_list[tile_id].buffer.read(mem_addr + i * dmem_word_len))
                self.tile_list[tile_id].load(operands[0], vec_data)

            elif op_code == "dot":
                I_out = self.tile_list[tile_id].dot(IPU_id, operands[0], operands[1], operands[2])
                # print(f"I_out = {np.array(I_out).shape}")
                result.append(I_out)
            elif op_code == "store":

                mem_addr = int(operands[1], 16)  # 在数据内存中的地址，16进制转成10进制
                vec_width = int(operands[2], 16)
                vec_data = self.tile_list[tile_id].store(operands[0], vec_width)

                for i in range(vec_width):
                    if len(operands[1]) == 8:
                        # 地址指向数据内存
                        self.dataMemory.write(mem_addr + i * dmem_word_len, vec_data[i])
                    else:
                        # 地址指向缓存
                        self.tile_list[tile_id].buffer.write(mem_addr + i * dmem_word_len, vec_data[i])
            self.pc.step()
        print("Execution finished!")
        return np.array(result)
