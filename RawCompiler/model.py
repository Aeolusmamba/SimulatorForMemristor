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
from data_converter import ConvDataConverter, LinearDataConverter
import copy


class ModelImpl():
    # convertedW = None
    chip = Chip(id=0)
    chipController = Controller()
    instGenerator = InstructionGenerator()
    reg_allocator = RegAllocator()
    weightConverter = WeightConverter()
    convDataConverter = ConvDataConverter()
    linearDataConverter = LinearDataConverter()
    layer_params = {}
    layer_size = 0
    base_size = 256
    layer_info = {}
    converted_Xshape = {}
    instruction = {}
    istart_addr = {}
    iend_addr = {}

    def __init__(self, file_path):
        if not os.path.exists(file_path):
            print("ERROR: File {} does not exist".format(file_path))
            sys.exit(1)
        self.file_path = file_path

    # 输入数据分配，编译时进行
    def assignInputResource(self, first_input: Variable, layer_params, layer_i, layer_info,
                            first_layer=False):

        self.converted_Xshape[layer_i] = self.chip.assignInput2Mem(first_input, layer_i, layer_params, layer_info, first_layer)

    def assignOutputResource(self, batch_size, layer_params: dict, layer_info):  # 预分配最终的输出特征图的内存地址
        """
        param input_shape: im2col之后的shape
        :param p_addrs:
        :param W_shape: converted之后的shape
        :param input_addrs: input的地址，output和input放在一个tile的rf中
        :return:
        """
        if layer_info["type"] == "conv":
            out_channel, in_channel, K_h, K_w = layer_params["W_shape"]
            in_c, h, w = layer_params["input_shape"]
            if isinstance(layer_params["stride"], tuple):
                stride_h, stride_w = layer_params["stride"]
            else:
                stride_h = stride_w = layer_params["stride"]
            if isinstance(layer_params["padding"], tuple):
                padding_h, padding_w = layer_params["padding"]
            else:
                padding_h = padding_w = layer_params["padding"]

            y_h = (h - K_h + 2 * padding_h) // stride_h + 1
            y_w = (w - K_w + 2 * padding_w) // stride_w + 1
            output_shape = [batch_size, out_channel, y_h, y_w]  # [64, 128, 28, 28]
        else:
            output_shape = [batch_size, layer_params["W_shape"]]
        self.chip.assignOutput2Mem(output_shape, layer_info)

    # deploy weight to IPU，编译时进行
    def deployWeight(self, W_shape, layer_info: dict, layer_i):
        addrs = layer_info["addrs"]
        XBar_hshape = layer_info.get('XBar_hshape', None)
        XBar_wshape = layer_info.get('XBar_wshape', None)

        if layer_info["type"] == "conv":
            out_channel, in_channel, K_h, K_w = W_shape
            W_XBar_shape = [in_channel * K_h * K_w, out_channel]
        else:
            W_XBar_shape = W_shape

        W_XBar = Variable(W_XBar_shape, name="W_XBar_" + str(layer_i))
        W_XBar.data = np.random.standard_normal(W_XBar_shape)  # make weight

        if XBar_hshape is not None and XBar_wshape is not None:  # medium/large scale
            start_i = 0
            for i in range(len(XBar_hshape)):
                start_j = 0
                for j in range(len(XBar_wshape)):
                    sub_w = W_XBar.data[start_i: start_i + XBar_hshape[i], start_j: start_j + XBar_wshape[j]]
                    block_idx = i * len(XBar_wshape) + j
                    if block_idx >= len(addrs):
                        raise Exception("Missing addresses after partitioning.")
                    addr = addrs[block_idx]
                    sub_W = Variable(list(sub_w.shape), name=f'sub_W_{block_idx}', scope=W_XBar.name)
                    sub_W.data = sub_w
                    tile_id, IPU_id = addr.split("_")
                    # print(f"tile_id: {tile_id}      IPU_id: {IPU_id}")
                    tile_id, IPU_id = int(tile_id), int(IPU_id)
                    self.chip.deployWeight(tile_id, IPU_id, sub_W)
                    start_j += XBar_wshape[j]
                start_i += XBar_hshape[i]
        else:  # small scale
            for addr in addrs:
                tile_id, IPU_id = addr.split("_")
                # print(f"tile_id: {tile_id}      IPU_id: {IPU_id}")
                tile_id, IPU_id = int(tile_id), int(IPU_id)
                self.chip.deployWeight(tile_id, IPU_id, W_XBar)

    def reshapeX(self, X: Variable, layer_params, layer_info):
        W_shape = layer_params["W_shape"]
        converted_X = None
        if layer_params["op_code"] == "conv":
            W_height = W_shape[1] * W_shape[2] * W_shape[3]  # height of logical W
            W_width = W_shape[0]  # width of logical W
            stride = layer_params['stride']
            padding = layer_params['padding']
            if "act_rows" in layer_info and "act_cols" in layer_info:
                # case 1，小规模卷积层
                if layer_info["act_cols"] > W_width:
                    replica_num = layer_info.get("replica_num", None)
                    if replica_num is None:
                        raise Exception("Replica_num does not exist!")
                    # if W_shape[1] == 1:
                    #     # 单通道，对角线重叠复制，需对输入数据并行组合
                    #     converted_X = self.convDataConverter.getConvertedDataOverlap(X, W_shape,
                    #                                                                      stride, padding,
                    #                                                                      replica_num)
                    #     # 某些极端情况下，组合后的data可能不规则，这里暂时不考虑这些复杂的情况，只关注理想的规则状态（即X的完整一行形成一列）
                    # else:
                    #     # 多通道，对角线非重叠复制，需对输入数据并行组合
                    converted_X = self.convDataConverter.getConvertedDataRep(X, W_shape,
                                                                                     stride,
                                                                                     padding, replica_num)
                else:
                    # 无复制
                    converted_X = self.convDataConverter.getConvertedData(X, W_shape, stride,
                                                                              padding)
            elif "XBar_hshape" in layer_info and "XBar_wshape" in layer_info:
                # case 3，中大规模卷积层，分块处理
                XBar_hshape = layer_info["XBar_hshape"]  # 根据W子矩阵在阵列中的height和width来切分X data
                XBar_wshape = layer_info["XBar_wshape"]
                converted_X = self.convDataConverter.getConvertedDataBlock(X, W_shape, stride,
                                                                               padding, XBar_hshape, XBar_wshape)

        elif layer_params["op_code"] == "linear":
            if "act_rows" in layer_info and "act_cols" in layer_info:
                # case 2，小规模线性层，无复制，无分块
                converted_X = self.linearDataConverter.getConvertedData(X)
            elif "XBar_hshape" in layer_info and "XBar_wshape" in layer_info:
                # case 3，中大规模线性层，分块处理
                XBar_hshape = layer_info["XBar_hshape"]  # 根据W子矩阵在阵列中的height来切分X data
                converted_X = self.linearDataConverter.getConvertedDataBlock(X, XBar_hshape)

        return converted_X

    # generate assembly instructions (string type) for each layer
    def generateAssemblyInstruction(self, layer_params, input_shape: list, layer_i, last_layer=False):
        """
        :param op_code:
        :param input_shape: im2col之后的shape
        :param input_addrs:
        :return:
        """
        inst_list = []
        batch_size = input_shape[0]
        op_code = layer_params["op_code"]
        act = layer_params["act"]

        if op_code == "conv":

            V_num = input_shape[1]  # 输入电压向量的个数同时也是计算的分组数，如784, == I_num
            V_depth = input_shape[2]
            I_depth = layer_params["W_shape"][0]
            addrs = self.layer_info[layer_i]["addrs"]
            XBar_hshape, XBar_wshape = self.layer_info[layer_i].get("XBar_hshape", None), self.layer_info[layer_i].get("XBar_wshape", None)

            for i in range(batch_size):
                for v_i in range(V_num):
                    tile_id, IPU_id = addrs[v_i % len(addrs)].split("_")  # 如果是分块后的权重，会有多个地址，将分块后的向量依次输入
                    tile_id, IPU_id = int(tile_id), int(IPU_id)
                    tile_x = self.chip.tile_list[tile_id]
                    xxx = tile_x.input_indexReg[layer_i]
                    input_index_reg_addrs = self.chip.tile_list[tile_id].input_indexReg[layer_i]  # 该层的输入索引寄存器地址列表
                    output_index_reg_addrs = self.chip.tile_list[tile_id].output_indexReg[layer_i]  # 该层的输出索引寄存器地址列表
                    # 找到这些index regs，取出值，得到掩码，load进入掩码指示的data_reg
                    mask = ""
                    for index_reg_addr in input_index_reg_addrs:
                        mask += self.chip.tile_list[tile_id].rf.index_reg[int(index_reg_addr, 2)].read()  # 连成一个256位的掩码
                    mask = list(mask)
                    input_data_regs = []  # input data的data reg是固定复用的，该临时列表用于存储数据寄存器的下标
                    for j in range(len(mask)):
                        if mask[j] == '1':
                            input_data_regs.append(j)
                    # 找到输出电流向量的索引寄存器，得到掩码
                    mask = ""
                    for index_reg_addr in output_index_reg_addrs:
                        mask += self.chip.tile_list[tile_id].rf.index_reg[int(index_reg_addr, 2)].read()  # 连成一个256位的掩码
                    mask = list(mask)
                    output_data_regs = []  # output data的data reg是固定复用的，用于存储数据寄存器的下标
                    for j in range(len(mask)):
                        if mask[j] == '1':
                            output_data_regs.append(j)
                    row_index_reg_addrs = self.chip.tile_list[tile_id].row_indexReg[IPU_id]

                    mem_data_depth = self.chip.dataMemory.data_depth
                    word_addr_offset = mem_data_depth // 8  # 每个字的地址偏移量

                    if layer_i == 0:
                        # 将input data load进相应的数据寄存器中
                        # SIMD格式，假定向量寄存器和数据在内存中都是连续的
                        input_start_addr = self.chip.dmem_input_table[i]  # 该输入样本在数据内存中的起始地址
                        mem_j_addr = input_start_addr + v_i * V_depth * word_addr_offset
                        rd = hex(input_data_regs[0])[2:].zfill(3)  # 3位十六进制寄存器地址
                        mem_j_addr = hex(mem_j_addr)[2:].zfill(8)  # 8位十六进制存储器地址
                        vec_width = hex(V_depth)[2:].zfill(3)  # 3位十六进制向量长度参数
                        inst = "load " + rd + " " + mem_j_addr + " " + vec_width
                        inst_list.append(inst)
                    else:
                        # 从本地buffer中进行读取数据
                        # 这里load进来的数据假定已经提前通过软件的方法im2col好了
                        input_start_addr = self.chip.tile_list[tile_id].layer_i_ptr[layer_i]
                        buffer_j_addr = input_start_addr + i * v_i * V_depth * word_addr_offset
                        rd = hex(input_data_regs[0])[2:].zfill(3)
                        buffer_j_addr = hex(buffer_j_addr)[2:].zfill(5)  # 5位十六进制存储器地址
                        vec_width = hex(V_depth)[2:].zfill(3)
                        inst = "load " + rd + " " + buffer_j_addr + " " + vec_width
                        inst_list.append(inst)

                    inst = "dot " + hex(int(row_index_reg_addrs[0], 2))[2:].zfill(3) + " " + hex(
                        int(input_index_reg_addrs[0], 2))[2:].zfill(3) \
                           + " " + hex(int(output_index_reg_addrs[0], 2))[2:].zfill(3)
                    inst_list.append(inst)

                    if len(addrs) == 1 or (len(addrs) > 1 and len(XBar_hshape) == 1):
                        if act is not None:
                            inst_list.append(self.act_inst(act, output_data_regs[0], I_depth))

                    else:  # 权重有纵向分块
                        current_loc = v_i % len(addrs)  # 当前向量所处的块编号（从0开始，行优先排列）
                        current_hloc, current_wloc = current_loc // len(XBar_wshape), current_loc % len(XBar_wshape)

                        if current_hloc != 0:
                            inst_list.append(self.add_vec(output_data_regs[0], output_data_regs[0],
                                                          self.chip.tile_list[tile_id].mov_reg[layer_i][0],
                                                          I_depth))

                        if current_hloc != len(XBar_hshape) - 1:
                            next_addr = (current_hloc + 1) * len(XBar_wshape) + current_wloc  # 下一行同一列的地址
                            next_tile, next_IPU = addrs[next_addr].split("_")
                            next_tile, next_IPU = int(next_tile), int(next_IPU)
                            inst_list.append(self.block_mv(output_data_regs[0],
                                                           self.chip.tile_list[next_tile].mov_reg[layer_i][0],
                                                           I_depth))
                        else:  # 最后一行
                            # 由于激活函数的非线性，故需要先累加再求激活值
                            if act is not None:
                                inst_list.append(self.act_inst(act, output_data_regs[0], I_depth))

                    if last_layer:
                        # SIMD格式，假定向量寄存器和数据在内存中都是连续的
                        output_start_addr = self.chip.dmem_output_table[i]  # 该输出样本在数据内存中的起始地址
                        mem_j_addr = output_start_addr + v_i * I_depth * word_addr_offset
                        mem_j_addr = hex(mem_j_addr)[2:].zfill(8)  # 8位十六进制存储器地址
                        rs = hex(output_data_regs[0])[2:].zfill(3)  # 3位十六进制寄存器地址
                        vec_width = hex(I_depth)[2:].zfill(3)  # 3位十六进制向量长度参数
                        inst = "store " + rs + " " + mem_j_addr + " " + vec_width
                        inst_list.append(inst)
                    else:
                        # 将中间层的输出向量保存到相应的（下一层）tile buffer中
                        next_layer = layer_i + 1
                        next_addrs = self.layer_info[next_layer]["addrs"]
                        tmp_tile = None

                        # 根据下一层的权重信息来进行零填充
                        # next_padding = next_params['padding']
                        # padding_h = padding_w = next_inc = next_h = next_w = None

                        # if next_padding is not None:
                        #     if isinstance(next_padding, tuple):
                        #         padding_h, padding_w = next_padding
                        #     else:
                        #         padding_h = padding_w = next_padding
                        #
                        #     next_inc, next_h, next_w = next_params['input_shape']

                        for addr in next_addrs:
                            next_tile, next_IPU = addr.split("_")
                            next_tile, next_IPU = int(next_tile), int(next_IPU)
                            if next_tile != tmp_tile:
                                # 如果下一层有多个地址，避免重复保存到同一个tile的buffer中
                                # 且意味着下一层权重分块，将“全部”输入数据保存到每一个存储分块矩阵的tile中
                                tmp_tile = next_tile

                                output_start_addr = self.chip.tile_list[next_tile].layer_i_ptr[next_layer]
                                bf_ptr = output_start_addr + v_i * I_depth * word_addr_offset
                                I_len = hex(I_depth)[2:].zfill(3)

                                # if padding_h is not None:
                                #     rs = self.chip.tile_list[next_tile].rf.zero_reg_id
                                #     rs = hex(rs)[2:].zfill(3)
                                #
                                #     if v_i == 0 or v_i == V_num - 1:
                                #         # 在首行前面或末行后面添加p行填充
                                #         zero_I = next_w * padding_h + 2 * padding_w * padding_h
                                #         for z in range(zero_I):
                                #             buffer_j_addr = hex(bf_ptr)[2:].zfill(5)
                                #             inst = "store " + rs + " " + buffer_j_addr + " " + I_len
                                #             inst_list.append(inst)
                                #             bf_ptr += I_depth * self.chip.tile_list[next_tile].buffer.data_depth // 8
                                #
                                #     if v_i % next_w == 0:  # 每行的前导0
                                #         buffer_j_addr = hex(bf_ptr)[2:].zfill(5)
                                #         inst = "store " + rs + " " + buffer_j_addr + " " + I_len
                                #         inst_list.append(inst)
                                #         bf_ptr += I_depth * self.chip.tile_list[next_tile].buffer.data_depth // 8

                                buffer_j_addr = hex(bf_ptr)[2:].zfill(5)  # 5位十六进制缓存地址
                                rs = hex(output_data_regs[0])[2:].zfill(3)
                                inst = "store " + rs + " " + buffer_j_addr + " " + I_len
                                inst_list.append(inst)

                                # bf_ptr += I_depth * self.chip.tile_list[next_tile].buffer.data_depth // 8
                                # if padding_h is not None and (v_i + 1) % next_w == 0:  # 每行的后导0
                                #     buffer_j_addr = hex(bf_ptr)[2:].zfill(5)
                                #     inst = "store " + rs + " " + buffer_j_addr + " " + I_len
                                #     inst_list.append(inst)
                                #     bf_ptr += I_depth * self.chip.tile_list[next_tile].buffer.data_depth // 8
                                # self.chip.tile_list[next_tile].layer_i_ptr2[next_layer] = bf_ptr


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
            updateOperation = UpdateOperation()  # update W
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

    def act_inst(self, act, vstart_addr, v_len):
        return act + f" {hex(vstart_addr)[2:].zfill(3)} {hex(v_len)[2:].zfill(3)}"

    def add_vec(self, dst, src1, src2, v_len):
        return "add " + f"{hex(dst)[2:].zfill(3)} {hex(src1)[2:].zfill(3)} {hex(src2)[2:].zfill(3)} " \
                        f"{hex(v_len)[2:].zfill(3)}"

    def block_mv(self, src, dst, v_len):
        return "mov " + f"{hex(src)[2:].zfill(3)} {hex(dst)[2:].zfill(3)} {hex(v_len)[2:].zfill(3)}"

    def fileInterpreter(self, file_path):
        print(f"Interpreting file: {file_path} ...")
        df = pd.read_csv(file_path)
        max_flag = False
        average_flag = False
        for i, row in df.iterrows():
            # for each layer i
            if row['kernel height'] == 1 and row['kernel width'] == 1:  # linear layer
                layer_params = {}
                layer_params["op_code"] = "linear"
                W_shape = [row['IFM channel depth'], row['kernel depth']]
                layer_params['W_shape'] = W_shape
                layer_params['input_shape'] = row['IFM channel depth']
                self.layer_params[i] = layer_params
                self.layer_size += 1
            elif max_flag:
                self.layer_size += 1
            elif average_flag:
                self.layer_size += 1
            else:  # which is a conv layer
                kernel_shape = [row['kernel depth'], row['IFM channel depth'], row['kernel height'],
                                row['kernel width']]
                input_shape = [row['IFM channel depth'], row['IFM height'], row['IFM width']]
                stride = row['stride']
                if pd.isna(stride):
                    stride = 1
                elif isinstance(stride, str):  # 2,1
                    if "," in stride:
                        stride = tuple(map(int, stride.split(",")))
                    else:
                        stride = int(stride)

                padding = row['padding']
                if pd.isna(padding):
                    padding = 0
                elif isinstance(padding, str):  # 1,2
                    padding = tuple(map(int, padding.split(",")))

                layer_params = {}
                layer_params["op_code"] = "conv"
                layer_params["W_shape"] = kernel_shape
                layer_params['input_shape'] = input_shape
                layer_params['stride'] = stride
                layer_params['padding'] = padding

                # activation
                if not pd.isna(row['activation']) and row['activation'].lower() == 'relu':
                    layer_params["act"] = "relu"
                elif pd.isna(row['activation']):
                    layer_params["act"] = None

                self.layer_params[i] = layer_params
                self.layer_size += 1

        print("Interpreting result: ")
        for i in range(len(self.layer_params)):
            print(f"layer {i}: ", self.layer_params[i])

    def initLinearSScale(self, layer_i, W_shape):
        return self.chip.initSScale(layer_i, W_shape, type="linear")

    def initLinearMScale(self, layer_i, W_shape):
        return self.chip.initMScale(layer_i, W_shape, type="linear")

    def initLinearLScale(self, layer_i, W_shape):
        return self.chip.initLScale(layer_i, W_shape, type="linear")

    def initConvSScale(self, layer_i, K_shape):
        return self.chip.initSScale(layer_i, K_shape, type="conv")

    def initConvMScale(self, layer_i, K_shape):
        return self.chip.initMScale(layer_i, K_shape, type="conv")

    def initConvLScale(self, layer_i, K_shape):
        return self.chip.initLScale(layer_i, K_shape, type="conv")

    def initLayerResource(self):
        # 根据layer_size和layer的大小来初始化RRAM存储单元
        # 如果各层权重未全部放入，可以将其初始化到local buffer中
        # try 1: 从前往后分配，中间无流水线

        print(f"Initializing layer resources...")

        for i in range(self.layer_size):
            W_shape = self.layer_params[i]['W_shape']
            if self.layer_params[i]['op_code'] == 'linear':
                if W_shape[0] < self.base_size and W_shape[1] < self.base_size:
                    # Small Scale
                    layer_info = self.initLinearSScale(i, W_shape)
                    self.layer_info[i] = layer_info
                elif W_shape[0] < self.base_size * 64 and W_shape[1] < self.base_size * 64:  # 能映射到32个tile中
                    # Medium Scale
                    layer_info = self.initLinearMScale(i, W_shape)
                    self.layer_info[i] = layer_info
                elif W_shape[0] < self.base_size * 512 and W_shape[1] < self.base_size * 512:  # 能映射到256个tile（1个chip）中
                    # Large Scale
                    layer_info = self.initLinearLScale(i, W_shape)
                    self.layer_info[i] = layer_info

            elif self.layer_params[i]['op_code'] == 'conv':
                out_channel, in_channel, K_h, K_w = W_shape
                k_shape = copy.deepcopy(W_shape)
                W_height = in_channel * K_h * K_w
                W_width = out_channel
                stride = self.layer_params[i]['stride']
                if isinstance(stride, tuple):
                    S_x, S_y = stride
                else:
                    S_x = S_y = stride
                k_shape.append(S_x)
                k_shape.append(S_y)
                if W_height < self.base_size and W_width < self.base_size:
                    # Small Scale
                    layer_info = self.initConvSScale(i, k_shape)
                    self.layer_info[i] = layer_info
                elif W_height < self.base_size * 64 and W_width < self.base_size * 64:
                    # Medium Scale
                    layer_info = self.initConvMScale(i, k_shape)
                    self.layer_info[i] = layer_info
                elif W_height < self.base_size * 512 and W_width < self.base_size * 512:
                    # Large Scale
                    layer_info = self.initConvLScale(i, k_shape)
                    self.layer_info[i] = layer_info

        for layer in range(self.layer_size):
            addrs = self.layer_info[layer]["addrs"]
            for addr in addrs:
                tile_id, IPU_id = addr.split("_")
                tile_id, IPU_id = int(tile_id), int(IPU_id)
                print(f"W of layer {layer} is deployed into tile {tile_id}, IPU {IPU_id}")
                replica_num = self.layer_info[layer].get("replica_num", None)
                if replica_num is not None:
                    print(f"Small scale W is replicated {replica_num} times!")


    def storeInstructions(self, inst_list):  # store instructions to InstructionMemory
        return self.chip.storeInstructions(inst_list)

    def assignX2Buffer(self, convertedX: Variable, addrs, layer_i):
        for addr in addrs:
            tile_id, IPU_id = addr.split("_")
            tile_id, IPU_id = int(tile_id), int(IPU_id)
            self.chip.tile_list[tile_id].assignX2Buffer(convertedX, layer_i)

    def assignX2Mem(self, result_X: Variable):
        self.chip.assignX2Mem(result_X)

    def inference(self, input_variable: Variable):
        self.fileInterpreter(self.file_path)
        self.initLayerResource()
        batch_size = input_variable.shape[0]
        first_input = self.reshapeX(input_variable, self.layer_params[0], self.layer_info[0])
        result = None

        for layer in range(self.layer_size):
            layer_params = self.layer_params[layer]
            layer_info = self.layer_info[layer]
            W_shape = layer_params["W_shape"]
            # 实际部署权重
            self.deployWeight(W_shape, layer_info, layer_i=layer)

            self.assignInputResource(first_input, layer_params, layer, layer_info,
                                         first_layer=True if layer == 0 else False)

            if layer == self.layer_size - 1:
                self.assignOutputResource(first_input.shape[0], layer_params, layer_info)

        for layer in range(self.layer_size):
            layer_params = self.layer_params[layer]

            # 当前这一层的指令可以由相应的tile去执行
            print("Generating instructions...")
            self.instruction[layer] = self.generateAssemblyInstruction(layer_params, self.converted_Xshape[layer], layer,
                                                         last_layer=True if layer==self.layer_size-1 else False)

        for layer in range(self.layer_size):

            self.istart_addr[layer], self.iend_addr[layer] = self.storeInstructions(self.instruction[layer])

            i = 0
            print(f"The first 50 instructions: ")
            for inst in self.instruction[layer][:50]:
                print("PC:  0x" + hex(i)[2:].zfill(8) + "  inst:  " + inst)
                i += 1
            print(f"Total number of instructions: {len(self.instruction[layer])}")

            print("Executing instructions\n......")
            result = self.chip.execute(self.instruction[layer], self.layer_info[layer]["addrs"])

            # print(f"Shape of the result: {result.shape}")

            if layer != self.layer_size -1:
                # im2col，并存入下一层的buffer中
                input_shape = self.layer_params[layer + 1]["input_shape"]
                if self.layer_params[layer]["op_code"] == "conv":
                    in_shape = [batch_size] + input_shape
                else:
                    in_shape = [batch_size, input_shape]
                result = result.reshape(tuple(in_shape))
                X = Variable(list(result.shape), name="input_layer_"+str(layer+1))
                X.data = result
                converted_X = self.reshapeX(X, self.layer_params[layer + 1], self.layer_info[layer + 1])
                next_addrs = self.layer_info[layer + 1]["addrs"]
                self.assignX2Buffer(converted_X, next_addrs, layer + 1)
            else:
                result_X = Variable(list(result.shape), name="result")
                result_X.data = result
                # self.assignX2Mem(result_X)

        return result


if __name__ == "__main__":
    # make some data
    data = np.random.normal(10, 1, (64, 1, 28, 28))
    # data = np.random.normal(10, 1, (1, 1, 7, 7))
    # print(f"input data: {data}")
    input_variable = Variable(list(data.shape), "test")  # 需要一个数据拆分模块，根据batch_size来拆分数据，现在假设input_variable是一个batch的数据
    input_variable.data = data
    model = ModelImpl(file_path="../NetWork7.csv")
    result = model.inference(input_variable)
    result = np.reshape(result, (64, 50, 15, 15))
    print(f"Result: {result}")
    print(f"Shape of result: {result.shape}")
