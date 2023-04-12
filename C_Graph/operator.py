import numpy as np
from variable import Variable, GLOBAL_VARIABLE_SCOPE


class Operator(object):
    def __init__(self, name, input_variables, output_variables):

        # init input check
        if GLOBAL_VARIABLE_SCOPE.has_key(name):
            raise Exception("Operator %s has already exists !" % name)

        if not isinstance(input_variables, Variable) and not isinstance(input_variables[0], Variable):
            raise Exception("Operator %s 's input_variables is not instance(or list) of Variable!")

        if not isinstance(output_variables, Variable) and not isinstance(output_variables[0], Variable):
            raise Exception("Operator %s 's output_variables is not instance(or list) of Variable!")

        # register in GLOBAL_OP_SCOPE
        self.name = name
        GLOBAL_VARIABLE_SCOPE[self.name] = self

        self.child = []
        self.parent = []

        # register for input Variable's child and output Variable's parents
        register_graph(input_variables, output_variables, self)

        self.wait_forward = True
        # self.wait_backward = not self.wait_forward

    def forward(self):
        pass
        # if self.wait_forward == True:
        #     1.check_parent_eval()
        #         for variable in self.parent:
        #             variable.eval()
        #     2.do forward_cal()
        #     3.set wait_forward()
        #         self.wait_forward = False
        # else:
        #     pass

    def backward(self):
        pass
        # if self.wait_forward == True:
        #     pass
        # else:
        #     1.check_child_diffeval()
        #         for variable in self.child:
        #             variable.diff_eval()
        #     2.do backward_cal()
        #     3.set wait forward()
        #         self.wait_forward=True
        #


def register_graph(input_variable, output_variable, operator):
    if isinstance(input_variable, Variable) and isinstance(output_variable, Variable):
        input_variable.child.append(operator.name)
        output_variable.parent.append(operator.name)
        operator.parent.append(input_variable.name)
        operator.child.append(output_variable.name)

    elif isinstance(input_variable, Variable) and len(output_variable) > 1:
        for output in output_variable:
            input_variable.child.append(operator.name)
            output.parent.append(operator.name)
            operator.parent.append(input_variable.name)
            operator.child.append(output.name)

    elif isinstance(output_variable, Variable) and len(input_variable) > 1:
        for input in input_variable:
            input.child.append(operator.name)
            output_variable.parent.append(operator.name)
            operator.parent.append(input.name)
            operator.child.append(output_variable.name)

    elif len(output_variable) > 1 and len(input_variable) > 1:
        for input in input_variable:
            input.child.append(operator.name)
            operator.parent.append(input.name)
        for output in output_variable:
            output.parent.append(operator.name)
            operator.child.append(output.name)

    else:
        raise Exception('Operator name %s input,output list error' % operator.name)
