class RegAllocator:
    reg_table = {}

    def allocate(self, count):  # 将数据抽象成类传入记录
        reg_list = []
        for i in range(1000000000, 1011111111):
            if str(i) not in self.reg_table and count != 0:
                reg_list.append(str(i))
                self.reg_table[str(i)] = 1  # TODO: 记录数据对象
                count -= 1
                if count == 0:
                    break
        if not reg_list:
            raise Exception("No idle register!")
        return reg_list


if __name__ == "__main__":
    reg_a = RegAllocator()
    reg_a.allocate(1)
