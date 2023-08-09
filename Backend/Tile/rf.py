from ..reg import Reg
import numpy as np

class RegisterFile:

    def __init__(self, data_depth=64, index_num=256, data_num=256):
        self.index_num = index_num
        self.data_num = data_num
        self.data_depth = data_depth
        self.row_num = 256  # row num
        self.index_reg = []
        self.data_reg = []
        for i in range(index_num):
            reg = Reg(id=i)
            self.index_reg.append(reg)
        for i in range(data_num):
            reg = Reg(id=i)
            self.data_reg.append(reg)

    def initDataIndexRegs(self, vector_size):
        # index reg的值是一个64位的掩码，需要4个regs来形成一个完整的掩码
        index_reg_addrs = []
        mask = ["0"] * self.data_num  # 掩码
        # 生成掩码
        for j in range(vector_size):
            data_reg_i = self.initDataReg()
            mask[data_reg_i] = "1"
        # print("mask=", mask)
        index_reg_num = self.data_num // self.data_depth  # 所需的索引寄存器数量
        index_temp = []
        # 分配空闲寄存器，保证保存一个掩码的n（4/8）个寄存器连续
        for i in range(len(self.index_reg)):
            if not self.index_reg[i].used:
                self.index_reg[i].used = True
                # print(f"idle index reg{i}")
                index_temp.append(i)
                index_reg_addr = bin(i)[2:].zfill(int(np.log2(self.index_num)))
                index_reg_addrs.append(index_reg_addr)
                index_reg_num -= 1
                if index_reg_num == 0:
                    break
        # 给寄存器赋值部分掩码
        for i in range(0, self.data_num - self.data_depth + 1, self.data_depth):
            sub_mask = ''.join(mask[i:i+self.data_depth])
            # print(f"index_temp[{i // self.data_depth}]=", index_temp[i // self.data_depth])
            self.index_reg[index_temp[i // self.data_depth]].write(sub_mask)
        return index_reg_addrs

    def initDataReg(self):
        for i in range(len(self.data_reg)):
            if not self.data_reg[i].used:
                self.data_reg[i].used = True
                return i

    def initRowIndexRegs(self, act_rows):

        mask = ['0'] * self.row_num
        for j in act_rows:
            mask[j] = '1'
        index_reg_num = self.row_num // self.data_depth
        index_reg_addrs = []
        index_temp = []
        # 分配空闲寄存器，保证保存一个掩码的n（4/8）个寄存器连续
        for i in range(len(self.index_reg)):
            if not self.index_reg[i].used:
                self.index_reg[i].used = True
                index_temp.append(i)
                index_reg_addr = bin(i)[2:].zfill(int(np.log2(self.index_num)))
                index_reg_addrs.append(index_reg_addr)
                index_reg_num -= 1
                if index_reg_num == 0:
                    break

        for i in range(0, self.row_num, self.data_depth):
            self.index_reg[index_temp[i // self.data_depth]].value = ''.join(mask[i:i+self.data_depth])

        return index_reg_addrs

    # 写数据寄存器
    def lw(self, dst_addr, data):
        dst_addr = int(dst_addr, 16)
        self.data_reg[dst_addr].write(data)

    # 读数据寄存器
    def sw(self, src_addr):
        src_addr = int(src_addr, 16)
        return self.data_reg[src_addr].read()

    # 写一组数据寄存器
    def load(self, dst_addr, vec_data):
        dst_addr = int(dst_addr, 16)
        for i in range(len(vec_data)):
            self.data_reg[dst_addr + i].write(vec_data[i])

    def store(self, src_addr, vec_width):
        src_addr = int(src_addr, 16)
        return [self.data_reg[src_addr + i].read() for i in range(vec_width)]

    def getMask(self, src):
        # 以src为起始地址，连续4个64 bit的index regs中存的是一个完整的掩码
        src = int(src, 16)
        mask = ""
        for i in range(4):
            mask += self.index_reg[src+i].read()
        return mask

    def getListOfValue(self, reg_index, group="index"):
        '''
        获得一组寄存器的值
        :param reg_index: 该组寄存器的地址
        :param group: 'index' group or 'data' group
        :return:
        '''
        if group == 'index':
            data_list = [self.index_reg[i].read() for i in reg_index]
        else:
            data_list = [self.data_reg[i].read() for i in reg_index]
        return data_list

    def setListOfValue(self, reg_index, vals: list, group='index'):
        '''
        写入一组寄存器的值
        :param reg_index: 该组寄存器的地址
        :param vals: 这一组值构成的列表
        :param group: 'index' group or 'data' group
        :return:
        '''
        if group == 'index':
            for i in range(len(reg_index)):
                self.index_reg[reg_index[i]].write(vals[i])
        else:
            for i in range(len(reg_index)):
                self.data_reg[reg_index[i]].write(vals[i])

    def schedule_1_src(self):
        reg0 = Reg()
        reg1 = Reg()
        return reg0, reg1

    def schedule_2_src(self):
        reg0 = Reg()
        reg1 = Reg()
        reg2 = Reg()
        return reg0, reg1, reg2
