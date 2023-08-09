class InstructionDecoder:

    def __init__(self):
        pass

    def decode(self, inst):
        # returns op_code and operands
        split_str = inst.split(" ")
        op_code = split_str[0]
        operands = split_str[1:]
        return op_code, operands
