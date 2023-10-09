import numpy as np


class Crossbar:

    def __init__(self, size):
        if isinstance(size, list):
            self.size = size
        elif isinstance(size, int):
            self.size = (size, size)
        else:
            raise Exception("Size must be int or list of ints!")
        self.conductance = np.zeros(self.size)  # actual crossbars
        self.row_mask = None  # conductance * row_mask == activated rows
        self.col_mask = None  # conductance * col_mask == activated cols

    def dot(self, row_mask, V):
        self.row_mask_ = row_mask  # 未被使用
        # print(f"dot conductance shape: {self.conductance.shape}")
        I_out = []
        for i in range(len(self.row_mask)):
            row_s, row_e = self.row_mask[i]
            col_s, col_e = self.col_mask[i]
            V_ = V[row_s:row_e+1]
            if V_.shape[0] != self.conductance[row_s:row_e+1, col_s:col_e+1].shape[0]:
                raise Exception(f"The shapes of V_{V_.shape} and conductance{self.conductance[row_s:row_e+1, col_s:col_e+1].shape} do not match")
            else:
                I = np.dot(V_, self.conductance[row_s:row_e+1, col_s:col_e+1])  # automatic broadcasting
                I_out.append(I)
        return np.reshape(I_out, (-1)).tolist()

    def mul(self, C):
        if C.shape != self.conductance.shape:
            raise Exception(f"The shapes of C({C.shape}) and conductance({self.conductance.shape}) do not match")
        else:
            return C * self.conductance

    def add(self, C):
        if C.shape != self.conductance.shape:
            raise Exception(f"The shapes of C({C.shape}) and conductance({self.conductance.shape}) do not match")
        else:
            return C + self.conductance

    def setRowMask(self, row_mask):
        self.row_mask = row_mask

    def setColMask(self, col_mask):
        self.col_mask = col_mask

    def read_conductance(self):
        return self.conductance

    def write_conductance(self, C):
        if self.row_mask is None or self.col_mask is None:
            raise Exception(f"This crossbar has not been initialized.")
        for i in range(len(self.row_mask)):
            row_start, row_end = self.row_mask[i]
            col_start, col_end = self.col_mask[i]
            if row_end-row_start+1 != C.shape[0] or col_end-col_start+1 != C.shape[1]:
                raise Exception(f"Cannot be written! The shape of (sub) conductance "
                                f"({(row_end-row_start+1, col_end-col_start+1)}) "
                                f"does not match the shape of C ({C.shape}).")
            self.conductance[row_start:row_end+1, col_start:col_end+1] = C
            # print(f"conductance.shape: {self.conductance.shape}")

    def read_cell(self, i, j):
        if i > self.conductance.shape[0] or j > self.conductance.shape[1]:
            raise Exception(f"i {i} or j {j} exceed the maximum shape of conductance -- {self.conductance.shape}")
        return self.conductance[i, j]

    def write_cell(self, i, j, c):
        if i > self.conductance.shape[0] or j > self.conductance.shape[1]:
            raise Exception(f"i {i} or j {j} exceed the maximum shape of conductance -- {self.conductance.shape}")
        self.conductance[i, j] = c

    # 初始化一个权重矩阵
    def initXBar(self, W_height, W_width):
        if self.row_mask is None or self.col_mask is None:
            self.row_mask = []
            self.col_mask = []
            self.row_mask.append([0, W_height-1])
            self.col_mask.append([0, W_width-1])
            act_rows = W_height  # 激活的行总数
            act_cols = W_width  # 激活的列总数


        else:  # 对角线“非重叠”复制
            last_row = self.row_mask[-1][1]
            last_col = self.col_mask[-1][1]
            self.row_mask.append([last_row+1, last_row+W_height])
            self.col_mask.append([last_col+1, last_col+W_width])
            act_rows = last_row + 1 + W_height
            act_cols = last_col + 1 + W_width

        return act_rows, act_cols

    # 对角线（非重叠）复制
    def initXBarRep(self, W_height, W_width, replica_num):
        self.row_mask = []
        self.col_mask = []
        row = act_rows = 0  # act_row：激活的行总数
        col = act_cols = 0  # act_col：激活的列总数
        for i in range(replica_num):
            self.row_mask.append([row, row + W_height - 1])
            self.col_mask.append([col, col + W_width - 1])
            act_rows = row + W_height
            act_cols = col + W_width
            row += W_height
            col += W_width
        return act_rows, act_cols

    # 对角线重叠复制
    def initXBarRepOverlap(self, W_height, W_width, replica_num, offset):
        self.row_mask = []
        self.col_mask = []
        row = act_rows = 0  # act_row：激活的行总数
        col = act_cols = 0  # act_col：激活的列总数
        for i in range(replica_num):
            self.row_mask.append([row, row+W_height-1])
            self.col_mask.append([col, col+W_width-1])
            act_rows = row + W_height
            act_cols = col + W_width
            row += offset
            col += W_width
        return act_rows, act_cols
