from ..Tile.tile import Tile
import numpy as np
from .inst_mem import InstructionMemory
from .inst_decoder import InstructionDecoder
from .data_mem import DataMemory
from .accumulator import Accumulator
from C_Graph.variable import Variable
from .pc import PC


class Chip:

    def __init__(self, id, tile_num=256):
        self.id = id
        self.tile_list = []
        self.instMemory = InstructionMemory()  # 4GB，32位地址
        self.inst_depth = 64  # 64位指令字长
        self.instDecoder = InstructionDecoder()
        self.pc = PC(inst_depth=64)
        self.dataMemory = DataMemory()  # 4GB，32位地址
        self.accumulator = Accumulator()
        self.dmem_ptr = 0  # 数据内存指针，按字节编制，32位地址，十六进制字符串形式存储（8位）
        self.imem_ptr = 0  # 指令内存指针，按字节编址，32位地址，十六进制字符串形式存储（8位）
        self.tile_usage = []  # tile使用表，tile_usage[i]=1 代表tile_id=i的tile被使用了
        self.used = False  # 当且仅当 tile_usage[i]=1 for i=0~n-1 时used=True
        self.dmem_input_table = {}  # data memory的输入数据地址索引表，键：sample的唯一标识（暂时为batch_i），值：该sample的起始地址
        self.dmem_output_table = {}  # data memory的输出数据地址索引表，键：sample的唯一标识（暂时为batch_i），值：该sample的起始地址
        self.imem_table = {}  # key: insts的PC word, value: inst mem中的（起始）地址
        for i in range(tile_num):
            tile = Tile(id=i)
            self.tile_list.append(tile)
        self.tile_usage = [0] * tile_num

    def assignInput2Mem(self, input_variable: Variable, W_shape: list):
        batch_size = input_variable.shape[0]
        out_channel, in_channel, K_h, K_w = W_shape
        tile_num = len(self.tile_list)
        tile_cnt = 0
        # IPU_num = len(self.tile_list[0].ipu_list) // 2  # Only 2 will be used as weight matrix
        IPU_cnt = 0
        addrs = []
        for i in range(batch_size):
            sample = input_variable.data[i]  # [784, 9]

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
            addr = str(tile_cnt) + "_" + str(IPU_cnt)
            addrs.append(addr)

            # 在相应的tile中，根据向量大小初始化索引寄存器的值
            self.initInputIndexRegs(tile_cnt, i, vector_size=input_variable.shape[2])
            self.dmem_input_table[i] = self.dmem_ptr

            # 每个样本连续存放在数据内存中，样本与样本之间可以不连续
            # print("storing input data...")
            # print(f"initial storing addr: ", self.dmem_ptr)
            for data in np.reshape(sample, -1):
                # print(self.dmem_ptr)
                self.dataMemory.write(self.dmem_ptr, data)  # write a word
                self.dmem_ptr += self.dataMemory.data_depth // 8  # 加上一个字的地址偏移，按字节编址
        self.input_addrs = addrs
        return addrs

    def initInputIndexRegs(self, tile_id, input_i, vector_size):
        self.tile_list[tile_id].initInputIndexRegs(input_i, vector_size)

    def initOutputIndexRegs(self, tile_id, input_i, vector_size):
        self.tile_list[tile_id].initOutputIndexRegs(input_i, vector_size)

    def deployWeight(self, tile_id, IPU_id, weight: Variable):
        act_rows = self.tile_list[tile_id].deployWeight(IPU_id, weight)  # activated rows list
        self.initRowIndexRegs(tile_id, act_rows)

    def initRowIndexRegs(self, tile_id, act_rows):
        self.tile_list[tile_id].initRowIndexRegs(act_rows)

    def assignOutput2Mem(self, output_shape, input_addrs):
        # 根据内存指针dmem_ptr开始分配
        batch_size, H_W, channel = output_shape
        sample_addr_offset = channel * H_W * self.dataMemory.data_depth // 8  # 每个样本的地址偏移量
        for i in range(batch_size):
            tile_id, IPU_id = input_addrs[i].split("_")
            tile_id, IPU_id = int(tile_id), int(IPU_id)
            self.initOutputIndexRegs(tile_id, i, vector_size=channel)
            self.dmem_output_table[i] = self.dmem_ptr  # 该样本的起始地址
            self.dmem_ptr += sample_addr_offset

    def storeInstructions(self, inst_list):
        inst_word_offset = self.inst_depth // 8  # 指令字地址偏移，64/8=8
        istart_addr = self.imem_ptr
        for i in range(len(inst_list)):
            self.imem_table[i] = self.imem_ptr
            self.instMemory.write(self.imem_ptr, inst_list[i])
            self.imem_ptr += inst_word_offset
        iend_addr = self.imem_ptr
        return istart_addr, iend_addr

    def execute(self, istart_addr, iend_addr):
        self.pc.reset(istart_addr)
        result = []
        tile_id, IPU_id = 0, 0
        dmem_word_len = self.dataMemory.data_depth // 8
        while self.pc.program_counter < iend_addr:
            inst = self.instMemory.read(self.pc.program_counter)
            if inst is None:
                raise Exception(f"Program counter {self.pc.program_counter} points to something unknown.")
            op_code, operands = self.instDecoder.decode(inst)
            sample_i = None
            if op_code == "load":
                dmem_addr = int(operands[1], 16)  # 在数据内存中的地址，16进制转成10进制
                vec_width = int(operands[2], 16)
                for key, value in self.dmem_input_table.items():
                    if dmem_addr == value:
                        sample_i = key
                        break
                if sample_i is not None:  # start of a sample
                    addr = self.input_addrs[sample_i]
                    tile_id, IPU_id = addr.split("_")
                    tile_id, IPU_id = int(tile_id), int(IPU_id)
                vec_data = []
                for i in range(vec_width):
                    vec_data.append(self.dataMemory.read(dmem_addr + i * dmem_word_len))
                self.tile_list[tile_id].load(operands[0], vec_data)
            elif op_code == "dot":
                I_out = self.tile_list[tile_id].dot(IPU_id, operands[0], operands[1], operands[2])
                result.append(I_out)
            elif op_code == "store":
                vec_width = int(operands[2], 16)
                vec_data = self.tile_list[tile_id].store(operands[0], vec_width)
                dmem_addr = int(operands[1], 16)  # 在数据内存中的地址，16进制转成10进制
                for i in range(vec_width):
                    self.dataMemory.write(dmem_addr + i * dmem_word_len, vec_data[i])
            self.pc.step()
        print("Execution finished!")
        return np.array(result)
