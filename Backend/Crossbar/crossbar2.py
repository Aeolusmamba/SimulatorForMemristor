import numpy as np
from Backend.backend import Backend
import functools

class Crossbar(object):
    Xbar_id = 0
    row_num = 256
    col_num = 256
    r_high = 1.0
    r_low = 0.2

    base = None
    # store the weights
    Vg0 = 1.0  # Initial SET gate voltage
    Vg_max = 1.6  # Max SET gate voltage
    Vg_min = 0.7  # SET gate voltage
    V_set = 2.5  # Fixed SET voltage
    V_reset = 1.7  # Fixed RESET voltage
    V_gate_reset = 5  # Fixed RESET gate voltage

    V_gate_set = None  # tuned parameter determines the final Gradient

    V_read = 0.2

    ratio_G_W = 100e-6
    ratio_Vg_G = 1 / 98e-6

    layer_ratio = [1, 1]

    # the array conductance for multiply reverse
    # update the value after weight update
    array_G = None

    # if true then plot
    draw = False

    # history; save history if true
    save = 0
    G_history = {}
    V_gate_history = {}
    V_reset_history = {}
    I_history = {}
    V_vec_history = {}
    # noise_model

    def __init__(self, base):
        self.base = base
        super(Crossbar, self).__init__()


    def initialize_weights(self):
        ok_args = {'draw', 'save'}
        defaults = [0, 0]

        self.V_gate_set = functools.reduce(lambda x: np.zeros(x.net_size) + self.Vg0, self.base.subs)
        # update the conductance for software back propagation
        self.base.update_subs('GND', self.V_reset, self.V_gate_reset)  # RESET pulse
        self.base.update_subs(self.V_set, 'GND', self.V_gate_set)  # SET pulse

        self.read_conductance('mode', 'fast')

    def read_conductance(self, *args):
        [self.array_G, fullG] = self.base.read_subs(args)

        # if self.draw:
            # figure(11);
            # subplot(3, 3, 1);
            # imagesc(fullG);
            # colorbar;
            # title('Conductance');
            # drawnow;

        if self.save:
            self.G_history[-1] = fullG


    def update(self, dWs):
        th_reset = 0

        th_set = 0

        nlayer = int(dWs)

        # if nlayer != numel(obj.base.subs)
        #     error('Wrong number of weight gradients');

        Vr = np.array(1, nlayer)
        Vg = np.array(1, nlayer)

        for layer in range(1, nlayer):
            dW = dWs[layer]

            dVg = self.ratio_Vg_G * self.ratio_G_W

            Vr[layer] = self.V_reset * (dVg < th_reset)
            Vg[layer] = self.V_gate_set[layer] + dVg
            Vg[Vg[layer] > self.Vg_max] = self.Vg_max
            Vg[Vg[layer] < self.Vg_min] = self.Vg_min

        p1 = self.base.update_subs('GND', Vr, self.V_gate_reset)  # RESET pulse
        p2 = self.base.update_subs(self.V_set, 'GND', Vg)  # SET pulse

        self.V_gate_set = Vg

        # update the conductance for software back propagation
        self.read_conductance('mode', 'fast')

        # if self.draw:
        #     subplot(3, 3, 2)
        #     imagesc(p1{2});colorbar
        #     title('Reset voltages')
        #
        #     subplot(3, 3, 3)
        #     imagesc(p2{3});colorbar
        #     title('Gate voltage')
        #     drawnow

        # if self.save:
        #     self.V_gate_history{end + 1} = p2{3}
        #     self.V_reset_history{end + 1} = p1{2}
        pass

    def multiply(self, vec, layer):
        self.check_layer(layer)

        # ASSUMPTION: vec is normalized to be max of 1
        voltage = np.array([vec, -vec]) * self.V_read
        current = self.base.subs[layer].read_current(voltage, 'gain', 2)

        output = current / self.V_read / self.ratio_G_W * self.layer_ratio[layer]

        # if self.draw >= 2:
        #     figure(11)
        #
        #     if layer <= 3:
        #         subplot(3, 3, layer + 3)
        #         plot(1: size(current, 1), current, 'o', 'MarkerSize', 1)
        #
        #         title(['I@' num2str(layer) '=' num2str(current(1))])
        #         # grid on; box on
        #         ylim([-2.4e-4, 2.4e-4])
        #         subplot(3, 3, layer + 6)
        #         plot(1: size(voltage, 1), voltage, 'o', 'MarkerSize', 1)
        #
        #         title(['V@' num2str(layer) '=' num2str(voltage(1))])
        #         # grid on; box on
        #         ylim([-0.3, 0.3])

        # if self.save:
        #     if layer == 1:
        #         self.V_vec_history{end + 1, layer} = voltage
        #         self.I_history{end + 1, layer} = current
        #     else
        #         self.V_vec_history{end, layer} = voltage
        #         self.I_history{end, layer} = current

    def check_layer(self, layer):
        pass
    # if layer > int(self.base.subs):
    #     error(['layer number should be less than ' num2str(numel(obj.W))])

    def multiply_reverse(self, vec, layer):
        self.check_layer(layer)

        G = self.array_G
        w = functools.reduce(lambda x: np.transpose(x[:  x.size() / 2,:]-x[x.size() / 2 + 1:,:]), G)

        output = w[layer].T * vec / self.ratio_G_W * self.layer_ratio[layer]  # w is tranposed compared to upper level algorithms
        pass

    def read_current(self, V_in):
        pass
