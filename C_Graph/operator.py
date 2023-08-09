import numpy as np
from C_Graph.variable import Variable, GLOBAL_VARIABLE_SCOPE


class Operator(object):
    def __init__(self, name, input_variables, output_variables):

        # initial input check
        if name in GLOBAL_VARIABLE_SCOPE:
            raise Exception("Operator %s has already exists !" % name)

        if not isinstance(input_variables, Variable) and not isinstance(input_variables[0], Variable):
            raise Exception("Operator %s 's input_variables is not instance(or list) of Variable!" % name)

        if not isinstance(output_variables, Variable) and not isinstance(output_variables[0], Variable):
            raise Exception("Operator %s 's output_variables is not instance(or list) of Variable!" % name)

        # register in GLOBAL_OP_SCOPE
        self.name = name
        GLOBAL_VARIABLE_SCOPE[self.name] = self

        self.child = []
        self.parent = []

        # register for input Variable's child and output Variable's parents
        register_graph(input_variables, output_variables, self)

        self.input_variable = input_variables
        self.output_variable = output_variables
        self.wait_forward = True
        # self.wait_backward = not self.wait_forward

    def forward(self, phase):
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

    def check_gradient(self):
        epsilon = 1e-5
        output_value = self.output_variable.data
        if isinstance(self.input_variable, list):
            self.input_variable[0].data -= epsilon
        else:
            self.input_variable.data -= epsilon
        self.wait_forward = True
        self.forward()
        output_1 = self.output_variable.data
        print("output_1: ", output_1)
        if isinstance(self.input_variable, list):
            self.input_variable[0].data += 2 * epsilon
        else:
            self.input_variable.data += 2 * epsilon
        self.wait_forward = True
        self.forward()
        output_2 = self.output_variable.data
        print("output_2: ", output_2)
        self.grad_approx = (output_2 - output_1) / (2 * epsilon)
        # restore value
        if isinstance(self.input_variable, list):
            self.input_variable[0].data -= epsilon
        else:
            self.input_variable.data -= epsilon
        self.wait_forward = True
        self.forward()
        self.output_variable.data = output_value


def register_graph(input_variable, output_variable, operator):
    if isinstance(input_variable, Variable) and isinstance(output_variable, Variable):
        input_variable.child.append(operator.name)
        output_variable.parent.append(operator.name)
        operator.parent.append(input_variable.name)
        operator.child.append(output_variable.name)

    elif isinstance(input_variable, Variable) and len(output_variable) > 1:
        input_variable.child.append(operator.name)
        operator.parent.append(input_variable.name)
        for output in output_variable:
            output.parent.append(operator.name)
            operator.child.append(output.name)

    elif isinstance(output_variable, Variable) and len(input_variable) > 1:
        output_variable.parent.append(operator.name)
        operator.child.append(output_variable.name)
        for input in input_variable:
            input.child.append(operator.name)
            operator.parent.append(input.name)

    elif len(input_variable) > 1 and len(output_variable) > 1:
        for input in input_variable:
            input.child.append(operator.name)
            operator.parent.append(input.name)
        for output in output_variable:
            output.parent.append(operator.name)
            operator.child.append(output.name)

    else:
        raise Exception('Operator name %s input, output list error' % operator.name)
