import numpy as np

class Evaluator():
    def compute_energy(self, in_channels: list, in_dims: list, out_channels: list, out_dims: list, kernel_sizes: list, C_size: int, device):
        if device == 'RRAM':
            xbar_ar = 1.76423
        elif device == 'SRAM':
            xbar_ar = 6721.089
        else:
            raise Exception("Unknown device type!")

        print(f"\nEvaluating energy with \"{device}\" device...\n")

        Tile_buff = 397
        Temp_Buff = 0.2
        Sub = 1.15E-6
        ADC = 2.03084

        Htree = 19.64 * 8  # 4.912*4 3.11E+6/30.*0.25
        # Include PE dependent HTree
        MUX = 0.094245
        mem_fetch = 4.64
        neuron = 1.274 * 4.0
        layer_size = len(out_channels)

        IPU_cycle_energy = []

        for i in range(layer_size):
            IPU_ar = kernel_sizes[i] * kernel_sizes[i] * xbar_ar + (C_size / 8) * (ADC + MUX)
            IPU_cycle_energy.append(Htree * mem_fetch + neuron + C_size / 8 * IPU_ar + \
                           (C_size / 8) * 16 * Sub + (C_size / 8) * Temp_Buff + Tile_buff)

        energy_layerwise = []
        total_energy = 0
        total_IPU_cycle = 0
        for i in range(layer_size):
            layer_IPU_cycle = np.ceil(out_channels[i] / C_size) * np.ceil(in_channels[i] * kernel_sizes[i] ** 2 / C_size) * (out_dims[i] ** 2)
            total_energy += layer_IPU_cycle * IPU_cycle_energy[i]
            total_IPU_cycle += layer_IPU_cycle
            energy_layerwise.append(layer_IPU_cycle * IPU_cycle_energy[i])
            print(f"layer_{i}: {energy_layerwise[i]} pJ")

        print(f"\ntotal energy: {total_energy} pJ")
        # return total_energy, energy_layerwise

    def compute_area(self, in_channels: list, in_dims: list, out_channels: list, out_dims: list, IPU_per_tile, n_tiles, kernel_sizes: list, C_size: int, device):
        if device == 'RRAM':
            xbar_ar = 26.2144
        elif device == 'SRAM':
            xbar_ar = 671.089
        else:
            raise Exception("Unknown device type!")

        print(f"\nEvaluating area with \"{device}\" device...\n")

        Tile_buff = 0.7391 * 64 * 128
        Temp_Buff = 484.643999
        Sub = 13411.41498
        ADC = 693.633

        Htree = 216830 * 2
        # Include PE dependent HTree
        MUX = 45.9
        layer_size = len(in_channels)

        IPU_ar = []
        tile_ar = []
        layer_area = []
        total_area = 0
        digital_area = []
        for i in range(layer_size):
            IPU_ar.append(kernel_sizes[i] * kernel_sizes[i] * xbar_ar + (C_size / 8) * (ADC + MUX))
            tile_ar.append((C_size / 8) * Sub * Temp_Buff + Tile_buff + IPU_per_tile * IPU_ar[i] + Htree)
            layer_area.append(tile_ar[i] * n_tiles + in_channels[i] * in_dims[i] ** 2 * 0.7391 * 22)
            total_area += layer_area[i]
            digital_area.append((Temp_Buff + Tile_buff + IPU_per_tile * IPU_ar[i]) * n_tiles + in_channels[i] * in_dims[i] ** 2 * 0.7391 * 22)
            print(f"layer {i}: digital area = {digital_area[i]} µm^2 ; total area = {layer_area[i]} µm^2")

        print(f"\ntotal area = {total_area} µm^2")






