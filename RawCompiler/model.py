import os
import sys
import numpy as np
import pandas as pd
from Backend.Chip.chip import Chip
from Backend.Chip.controller import Controller
from inst_gen import InstructionGenerator
from operations import *
from Backend.reg import Reg
from reg_allocate import RegAllocator
from C_Graph.variable import Variable, GLOBAL_VARIABLE_SCOPE
from C_Graph.operator import Operator
from weight_converter import WeightConverter
from data_converter import DataConverter


class ModelImpl():
    convertedW = None
    convertedData = None
    input_variable = None
    chip = Chip(id=0)
    chipController = Controller()
    instGenerator = InstructionGenerator()
    reg_allocator = RegAllocator()
    weightConverter = WeightConverter()
    dataConverter = DataConverter()
    layer_params = {}
    layer_size = 0

    def __init__(self, input_variable, file_path):
        self.input_variable = input_variable  # 需要一个数据拆分模块，根据batch_size来拆分数据，现在假设input_variable是一个batch的数据
        if not os.path.exists(file_path):
            print("ERROR: File {} does not exist".format(file_path))
            sys.exit(1)
        self.file_path = file_path

    # 输入数据分配，编译时进行
    def assignInputResource(self, W_size, stride, padding):
        self.convertedData = self.dataConverter.getConvertedData(self.input_variable, W_size, stride, padding)
        addrs = self.chip.assignInput2Mem(self.convertedData, W_size)  # 整个batch的数据被分配到的物理tile/IPU地址
        return addrs

    def assignOutputResource(self, input_shape: list, W_shape: list, input_addrs: list):  # 预分配输出特征图的内存地址
        """
        param input_shape: im2col之后的shape
        :param p_addrs:
        :param W_shape: converted之后的shape
        :param input_addrs: input的地址，output和input放在一个tile的rf中
        :return:
        """

        self.output_shape = (input_shape[0], input_shape[1], W_shape[0])  # 64*784*128
        self.chip.assignOutput2Mem(self.output_shape, input_addrs)

    # deploy weight to IPU，编译时进行
    def deployWeight(self, W, base_size, input_addrs:list):
        self.convertedW = self.weightConverter.getConvertedWeight(W, base_size)
        if len(self.convertedW) == 1:
            self.convertedW = self.convertedW[0]
            for addr in input_addrs:
                tile_id, IPU_id = addr.split("_")
                # print(f"tile_id: {tile_id}      IPU_id: {IPU_id}")
                tile_id, IPU_id = int(tile_id), int(IPU_id)
                self.chip.deployWeight(tile_id, IPU_id, self.convertedW)
        # else: # weight多于一个IPU的情况

    # generate assembly instructions (string type) for each layer
    def generateAssemblyInstruction(self, op_code, input_shape: list, input_addrs, last_layer=False):
        """
        :param op_code:
        :param input_shape: im2col之后的shape
        :param input_addrs:
        :return:
        """
        inst_list = []
        batch_size = input_shape[0]
        V_num = input_shape[1]  # 输入电压向量的个数同时也是计算的分组数，如784, == I_num
        V_depth = input_shape[2]

        I_depth = self.output_shape[2]

        if op_code == "conv":
            for i in range(batch_size):
                tile_id, IPU_id = input_addrs[i].split("_")
                tile_id, IPU_id = int(tile_id), int(IPU_id)
                input_index_reg_addrs = self.chip.tile_list[tile_id].input_indexReg[i]  # 该样本的输入索引寄存器地址列表
                output_index_reg_addrs = self.chip.tile_list[tile_id].output_indexReg[i]  # 该样本的输出索引寄存器地址列表
                # 找到这些index regs，取出值，得到掩码，load进入掩码指示的data_reg
                mask = ""
                for index_reg_addr in input_index_reg_addrs:
                    mask += self.chip.tile_list[tile_id].rf.index_reg[int(index_reg_addr, 2)].read()  # 连成一个256位的掩码
                mask = list(mask)
                input_data_regs = []  # input data的data reg是固定复用的，用于存储数据寄存器的下标
                for j in range(len(mask)):
                    if mask[j] == '1':
                        input_data_regs.append(j)
                if V_depth != len(input_data_regs):
                    raise Exception(f"V_depth {V_depth} does not equal to len(input_data_regs) {len(input_data_regs)}!")
                # 找到输出电流向量的索引寄存器，得到掩码
                mask = ""
                for index_reg_addr in output_index_reg_addrs:
                    mask += self.chip.tile_list[tile_id].rf.index_reg[int(index_reg_addr, 2)].read()  # 连成一个256位的掩码
                mask = list(mask)
                output_data_regs = []  # output data的data reg是固定复用的，用于存储数据寄存器的下标
                for j in range(len(mask)):
                    if mask[j] == '1':
                        output_data_regs.append(j)
                if I_depth != len(output_data_regs):
                    raise Exception(f"I_depth {I_depth} does not equal to len(output_data_regs) {len(output_data_regs)}!")
                row_index_reg_addrs = self.chip.tile_list[tile_id].row_indexReg

                input_start_addr = self.chip.dmem_input_table[i]  # 该输入样本在数据内存中的起始地址
                output_start_addr = self.chip.dmem_output_table[i]  # 该输出样本在数据内存中的起始地址
                mem_data_depth = self.chip.dataMemory.data_depth
                word_addr_offset = mem_data_depth // 8  # 每个字的地址偏移量

                for V_i in range(V_num):  # 按输入电压向量分组
                    # 将input data load进相应的数据寄存器中

                    # for j in range(V_depth):
                    #     mem_j_addr = input_start_addr + V_i * V_depth * word_addr_offset + j * word_addr_offset
                    #     # rd_bin = "11" + bin(input_data_regs[j])[2:]
                    #     rd = hex(input_data_regs[j])[2:].zfill(3)  # 3位十六进制寄存器地址
                    #     mem_j_addr = hex(mem_j_addr)[2:].zfill(8)  # 8位十六进制存储器地址
                    #     inst = "lw " + rd + " " + mem_j_addr
                    #     inst_list.append(inst)

                    # SIMD格式，假定向量寄存器和数据在内存中都是连续的
                    mem_j_addr = input_start_addr + V_i * V_depth * word_addr_offset
                    rd = hex(input_data_regs[0])[2:].zfill(3)  # 3位十六进制寄存器地址
                    mem_j_addr = hex(mem_j_addr)[2:].zfill(8)  # 8位十六进制存储器地址
                    vec_width = hex(V_depth)[2:].zfill(3)  # 3位十六进制向量长度参数
                    inst = "load " + rd + " " + mem_j_addr + " " + vec_width
                    inst_list.append(inst)

                    inst = "dot " + hex(int(row_index_reg_addrs[0], 2))[2:].zfill(3) + " " + hex(int(input_index_reg_addrs[0], 2))[2:].zfill(3) \
                           + " " + hex(int(output_index_reg_addrs[0], 2))[2:].zfill(3)
                    inst_list.append(inst)
                    # Version 1，将结果移动回data memory
                    # for j in range(I_depth):
                    #     mem_j_addr = output_start_addr + V_i * I_depth * word_addr_offset + j * word_addr_offset
                    #     mem_j_addr = hex(mem_j_addr)[2:].zfill(8)  # 8位十六进制存储器地址
                    #     # rs_bin = "11" + bin(output_data_regs[j])[2:]
                    #     rs = hex(output_data_regs[j])[2:].zfill(3)  # 3位十六进制寄存器地址
                    #     inst = "sw " + rs + " " + mem_j_addr
                    #     inst_list.append(inst)
                        # SIMD格式，假定向量寄存器和数据在内存中都是连续的
                    mem_j_addr = output_start_addr + V_i * I_depth * word_addr_offset
                    mem_j_addr = hex(mem_j_addr)[2:].zfill(8)  # 8位十六进制存储器地址
                    rs = hex(output_data_regs[0])[2:].zfill(3)  # 3位十六进制寄存器地址
                    vec_width = hex(I_depth)[2:].zfill(3)  # 3位十六进制向量长度参数
                    inst = "store " + rs + " " + mem_j_addr + " " + vec_width
                    inst_list.append(inst)

            return inst_list

        elif op_code == "linear":
            dotOperation = DotOperation()
            movOperation = MovOperation()
            # schedule registers
            src1_reg, src2_reg, dst_reg = self.reg_allocator.allocate(3)
            dotOperation.setSrc1Reg(src1_reg)
            dotOperation.setSrc2Reg(src2_reg)
            dotOperation.setDstReg(dst_reg)
            movOperation.setSrcReg(dst_reg)
            movOperation.setDstReg(src1_reg)
            dot = self.instGenerator.instGen(dotOperation)
            mov = self.instGenerator.instGen(movOperation)
            return dot, mov
        elif op_code == "max_pool":  # move to some digital circuit
            movOperation = MovOperation()
            src_reg, dst_reg = self.reg_allocator.allocate(2)
            movOperation.setSrcReg(src_reg)
            movOperation.setDstReg(dst_reg)
            mov = self.instGenerator.instGen(movOperation)
            return mov
        elif op_code == "de_conv":
            dotOperation1 = DotOperation()  # for the previous diff
            dotOperation2 = DotOperation()  # for gradient of W
            updateOperation = UpdateOperation() # update W
            movOperation = MovOperation()  # move the previous diff to the previous layer
            # schedule registers
            src1_reg, src2_reg, dst1_reg = self.reg_allocator.allocate(3)
            dotOperation1.setSrc1Reg(src1_reg)
            dotOperation1.setSrc2Reg(src2_reg)
            dotOperation1.setDstReg(dst1_reg)
            src3_reg, src4_reg, dst2_reg = self.reg_allocator.allocate(3)  # 需要再向上层抽象，比如由controller来schedule
            dotOperation2.setSrc1Reg(src3_reg)
            dotOperation2.setSrc2Reg(src4_reg)
            dotOperation2.setDstReg(dst2_reg)
            movOperation.setSrcReg(dst1_reg)
            movOperation.setDstReg(src1_reg)
            dot1 = self.instGenerator.instGen(dotOperation1)
            dot2 = self.instGenerator.instGen(dotOperation2)
            update = self.instGenerator.instGen(updateOperation)
            mov = self.instGenerator.instGen(movOperation)
            return dot1, dot2, update, mov
        elif op_code == "de_linear":
            dotOperation1 = DotOperation()  # for the previous diff
            dotOperation2 = DotOperation()  # for gradient of W
            updateOperation = UpdateOperation()  # update W
            movOperation = MovOperation()  # move the previous diff to the previous layer
            # schedule registers
            src1_reg, src2_reg, dst1_reg = self.reg_allocator.allocate(3)
            dotOperation1.setSrc1Reg(src1_reg)
            dotOperation1.setSrc2Reg(src2_reg)
            dotOperation1.setDstReg(dst1_reg)
            src3_reg, src4_reg, dst2_reg = self.reg_allocator.allocate(3)  # Compiler的Register allocator 感知寄存器分配情况
            dotOperation2.setSrc1Reg(src3_reg)
            dotOperation2.setSrc2Reg(src4_reg)
            dotOperation2.setDstReg(dst2_reg)
            movOperation.setSrcReg(dst1_reg)
            movOperation.setDstReg(src1_reg)
            dot1 = self.instGenerator.instGen(dotOperation1)
            dot2 = self.instGenerator.instGen(dotOperation2)
            update = self.instGenerator.instGen(updateOperation)
            mov = self.instGenerator.instGen(movOperation)
            return dot1, dot2, update, mov

    def fileInterpreter(self, file_path):
        df = pd.read_csv(file_path)
        max_flag = False
        average_flag = False
        for i, row in df.iterrows():
            # for each layer i
            if row['kernel height'] == 1 and row['kernel width'] == 1:  # linear layer
                self.layer_size += 1
            elif max_flag:
                self.layer_size += 1
            elif average_flag:
                self.layer_size += 1
            else:  # which is a conv layer
                kernel_shape = [row['kernel depth'], row['IFM channel depth'], row['kernel height'],
                                row['kernel width']]
                stride = row['stride']
                if isinstance(stride, str):  # 2,1
                    stride = tuple(map(int, stride.split(",")))
                padding = row['padding']
                if isinstance(padding, str):  # 1,2
                    padding = tuple(map(int, padding.split(",")))
                layer_params = {}
                layer_params["op_code"] = "conv"
                layer_params["kernel_shape"] = kernel_shape
                layer_params['stride'] = stride
                layer_params['padding'] = padding

                # activation
                if not pd.isna(row['activation']) and row['activation'].lower() == 'relu':
                    layer_params["act"] = "relu"

                self.layer_params[i] = layer_params
                self.layer_size += 1

    def storeInstructions(self, inst_list):  # store instructions to InstructionMemory
        return self.chip.storeInstructions(inst_list)

    def inference(self):
        self.fileInterpreter(self.file_path)
        for layer in range(self.layer_size):
            layer_params = self.layer_params[layer]
            W_size = layer_params["kernel_shape"]
            W = Variable(W_size, "test_W")
            W.data = np.random.standard_normal(W_size)  # make weight
            print(f"convolutional kernel W: {W.data}")
            input_addrs = self.assignInputResource(W_size, layer_params["stride"], layer_params["padding"])
            self.assignOutputResource(input_variable.shape, W_size, input_addrs)
            self.deployWeight(W, base_size=256, input_addrs=input_addrs)
            last_layer = True if layer==self.layer_size-1 else False
            inst_list = self.generateAssemblyInstruction(layer_params["op_code"], self.convertedData.shape, input_addrs, last_layer)
            istart_addr, iend_addr = self.storeInstructions(inst_list)
            # i=0
            # for inst in inst_list[:150]:
            #     print("PC:  0x" + hex(i)[2:].zfill(8) + "  inst:  " + inst)
            #     i += 1
            # print(f"Total number of instructions: {len(inst_list)}")

            result = self.chip.execute(istart_addr, iend_addr)

            print(f"Shape of the result: {result.shape}")
            # print(f"Result: {result.T.reshape((1, 3, 7, 7))}")
            print(f"Result: {result}")

if __name__ == "__main__":
    # make some data
    data = np.random.normal(10, 1, (64, 1, 28, 28))
    # data = np.random.normal(10, 1, (1, 1, 7, 7))
    print(f"input data: {data}")
    input_variable = Variable(list(data.shape), "test")
    input_variable.data = data
    model = ModelImpl(input_variable, file_path="../NetWork5.csv")
    model.inference()
