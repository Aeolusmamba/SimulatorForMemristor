import numpy as np
from C_Graph.variable import Variable
from ..Crossbar.crossbar import Crossbar
from ..reg import Reg
from .dac import DAC
from .adc import ADC
from .integrator import Integrator
from .sh import SampleHold
from .sa import ShiftAdder


class IPU(object):

    def __init__(self, id, type="W"):
        self.id = id
        self.type = type
        self.crossbar = Crossbar(256)
        self.used = False
        self.inputReg = Reg()
        self.outputReg = Reg()
        self.dac = DAC()
        self.adc = ADC()
        self.integrator = Integrator()
        self.sampleHold = SampleHold()
        self.shiftAdder = ShiftAdder()

    def get_crossbar(self, Xbar_id):
        return

    def deployWeight(self, weight: Variable):
        self.crossbar.write_conductance(weight.data)

    def writeToXbarMem(self, data_addr, data):
        return

    def readFromXbarMem(self, data_addr):
        return

    def dot(self, row_mask, V: list):
        V = np.array(V)
        return self.crossbar.dot(row_mask, V)
