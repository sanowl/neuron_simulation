import numpy as np
from scipy.integrate import solve_ivp
from numba import jit
import defusedxml.ElementTree

class IonChannel:
    def __init__(self, g_max, E_rev):
        self.g_max = g_max
        self.E_rev = E_rev

    def current(self, V, state, segment):
        raise NotImplementedError

    def state_derivatives(self, V, state):
        raise NotImplementedError

class SodiumChannel(IonChannel):
    def current(self, V, state, segment):
        m = state['Na_m'][segment]
        h = state['Na_h'][segment]
        return self.g_max * m**3 * h * (V - self.E_rev)

    def state_derivatives(self, V, state):
        alpha_m = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta_m = 4 * np.exp(-(V + 65) / 18)
        alpha_h = 0.07 * np.exp(-(V + 65) / 20)
        beta_h = 1 / (1 + np.exp(-(V + 35) / 10))

        dm = alpha_m * (1 - state['Na_m']) - beta_m * state['Na_m']
        dh = alpha_h * (1 - state['Na_h']) - beta_h * state['Na_h']

        return {'Na_m': dm, 'Na_h': dh}

class PotassiumChannel(IonChannel):
    def current(self, V, state, segment):
        n = state['K_n'][segment]
        return self.g_max * n**4 * (V - self.E_rev)

    def state_derivatives(self, V, state):
        alpha_n = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
        beta_n = 0.125 * np.exp(-(V + 65) / 80)

        dn = alpha_n * (1 - state['K_n']) - beta_n * state['K_n']

        return {'K_n': dn}

class LeakChannel(IonChannel):
    def current(self, V, state, segment):
        return self.g_max * (V - self.E_rev)

    def state_derivatives(self, V, state):
        return {}

class Synapse:
    def __init__(self, g_max, E_rev, tau):
        self.g_max = g_max
        self.E_rev = E_rev
        self.tau = tau

    def current(self, V, state, segment):
        s = state['syn_s'][segment]
        return self.g_max * s * (V - self.E_rev)

    def state_derivatives(self, t, state):
        ds = -state['syn_s'] / self.tau
        return {'syn_s': ds}

class Compartment:
    def __init__(self, length, diameter, channels):
        self.length = length
        self.diameter = diameter
        self.area = np.pi * diameter * length
        self.channels = channels

    def current(self, V, state):
        return sum(channel.current(V, state, 0) for channel in self.channels)

class NeuronMembrane:
    def __init__(self, morphology_file):
        self.morphology = self.load_morphology(morphology_file)
        self.channels = self.create_channels()
        self.synapses = self.create_synapses()
        self.compartments = self.create_compartments()
        
        self.state = self.initialize_state()

    def load_morphology(self, file):
        tree = defusedxml.ElementTree.parse(file)
        root = tree.getroot()
        
        morphology = {'soma': [], 'dendrites': [], 'axon': []}
        
        for cell in root.findall('.//cell'):
            for segment in cell.findall('.//segment'):
                comp_type = segment.get('type')
                length = float(segment.find('length').text)
                diameter = float(segment.find('diameter').text)
                
                if comp_type == 'soma':
                    morphology['soma'].append((length, diameter))
                elif comp_type == 'dendrite':
                    morphology['dendrites'].append((length, diameter))
                elif comp_type == 'axon':
                    morphology['axon'].append((length, diameter))
        
        return morphology

    def create_channels(self):
        return {
            'Na': SodiumChannel(g_max=120, E_rev=50),
            'K': PotassiumChannel(g_max=36, E_rev=-77),
            'Leak': LeakChannel(g_max=0.3, E_rev=-54.387)
        }

    def create_synapses(self):
        return [Synapse(g_max=0.1, E_rev=0, tau=2.0)]

    def create_compartments(self):
        compartments = []
        for comp_type in ['soma', 'dendrites', 'axon']:
            for length, diameter in self.morphology[comp_type]:
                compartments.append(Compartment(length, diameter, list(self.channels.values())))
        return compartments

    def initialize_state(self):
        num_compartments = len(self.compartments)
        state = {
            'V': np.full(num_compartments, -65.0),
            'Na_m': np.full(num_compartments, 0.05),
            'Na_h': np.full(num_compartments, 0.6),
            'K_n': np.full(num_compartments, 0.32),
            'syn_s': np.zeros(num_compartments),
            'Ca_i': np.full(num_compartments, 0.0001)
        }
        return state

    @jit(nopython=True)
    def dstate_dt(self, t, state_flat, I_ext):
        num_compartments = len(self.compartments)
        state = {
            'V': state_flat[:num_compartments],
            'Na_m': state_flat[num_compartments:2*num_compartments],
            'Na_h': state_flat[2*num_compartments:3*num_compartments],
            'K_n': state_flat[3*num_compartments:4*num_compartments],
            'syn_s': state_flat[4*num_compartments:5*num_compartments],
            'Ca_i': state_flat[5*num_compartments:]
        }

        dstate = {k: np.zeros_like(v) for k, v in state.items()}

        for i, comp in enumerate(self.compartments):
            I_channels = comp.current(state['V'][i], state)
            I_axial = 0
            
            C_m = 1.0  # µF/cm²
            dstate['V'][i] = (I_ext[i] - I_channels - I_axial) / (C_m * comp.area)

            for channel in comp.channels:
                channel_derivatives = channel.state_derivatives(state['V'][i], {k: v[i] for k, v in state.items()})
                for k, v in channel_derivatives.items():
                    dstate[k][i] = v

        for synapse in self.synapses:
            synapse_derivatives = synapse.state_derivatives(t, state)
            for k, v in synapse_derivatives.items():
                dstate[k] += v

        return np.concatenate([dstate[k] for k in state.keys()])

    def simulate(self, T, dt, I_ext_func):
        t_span = (0, T)
        t_eval = np.arange(0, T, dt)
        
        def wrapped_dstate_dt(t, y):
            I_ext = I_ext_func(t)
            return self.dstate_dt(t, y, I_ext)
        
        initial_state = np.concatenate([self.state[k] for k in self.state.keys()])
        
        solution = solve_ivp(
            wrapped_dstate_dt,
            t_span,
            initial_state,
            t_eval=t_eval,
            method='LSODA'
        )
        
        num_compartments = len(self.compartments)
        result_state = {
            'V': solution.y[:num_compartments],
            'Na_m': solution.y[num_compartments:2*num_compartments],
            'Na_h': solution.y[2*num_compartments:3*num_compartments],
            'K_n': solution.y[3*num_compartments:4*num_compartments],
            'syn_s': solution.y[4*num_compartments:5*num_compartments],
            'Ca_i': solution.y[5*num_compartments:]
        }
        
        return solution.t, result_state

    def apply_plasticity(self, simulation_result):
       
        for synapse in self.synapses:
            pre_spikes = self.detect_spikes(simulation_result.t, simulation_result.y[0])
            post_spikes = self.detect_spikes(simulation_result.t, simulation_result.y[-1])
            synapse.g_max += self.compute_stdp(pre_spikes, post_spikes)

    def detect_spikes(self, t, V, threshold=-20):
        return t[np.where(np.diff(np.sign(V - threshold)) > 0)[0]]

    def compute_stdp(self, pre_spikes, post_spikes, A_plus=0.005, A_minus=0.005, tau=20):
        delta_g = 0
        for pre in pre_spikes:
            for post in post_spikes:
                delta_t = post - pre
                if delta_t > 0:
                    delta_g += A_plus * np.exp(-delta_t / tau)
                else:
                    delta_g -= A_minus * np.exp(delta_t / tau)
        return delta_g

# Usage example:
if __name__ == "__main__":
    neuron = NeuronMembrane("neuron_morphology.xml")
    
    def I_ext_func(t):
        return np.array([10 if 100 <= t <= 300 else 0 for _ in range(len(neuron.compartments))])
    
    t, result = neuron.simulate(500, 0.1, I_ext_func)
    
    import matplotlib.pyplot as plt
    plt.plot(t, result['V'][0])
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.title('Soma Membrane Potential')
    plt.show()
