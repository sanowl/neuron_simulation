import numpy as np
from scipy.special import expit
from abc import ABC, abstractmethod

class Synapse(ABC):
    def __init__(self, pre_neuron, post_neuron, weight=0.5, delay=1.0):
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.weight = weight
        self.delay = delay
        self.last_spike_time = -np.inf

    @abstractmethod
    def compute_psp(self, t):
        pass

    @abstractmethod
    def update(self, t, dt):
        pass

class ExcitatorySynapse(Synapse):
    def __init__(self, pre_neuron, post_neuron, weight=0.5, delay=1.0, tau=5.0):
        super().__init__(pre_neuron, post_neuron, weight, delay)
        self.tau = tau

    def compute_psp(self, t):
        if t - self.last_spike_time < self.delay:
            return 0
        else:
            return self.weight * np.exp(-(t - self.last_spike_time - self.delay) / self.tau)

    def update(self, t, dt):
        if self.pre_neuron.has_spiked(t):
            self.last_spike_time = t

class InhibitorySynapse(Synapse):
    def __init__(self, pre_neuron, post_neuron, weight=0.5, delay=1.0, tau=10.0):
        super().__init__(pre_neuron, post_neuron, weight, delay)
        self.tau = tau

    def compute_psp(self, t):
        if t - self.last_spike_time < self.delay:
            return 0
        else:
            return -self.weight * np.exp(-(t - self.last_spike_time - self.delay) / self.tau)

    def update(self, t, dt):
        if self.pre_neuron.has_spiked(t):
            self.last_spike_time = t

class NMDASynapse(Synapse):
    def __init__(self, pre_neuron, post_neuron, weight=0.5, delay=1.0, tau_rise=2.0, tau_decay=100.0, mg_concentration=1.0):
        super().__init__(pre_neuron, post_neuron, weight, delay)
        self.tau_rise = tau_rise
        self.tau_decay = tau_decay
        self.mg_concentration = mg_concentration
        self.g = 0.0

    def compute_psp(self, t):
        V = self.post_neuron.V
        mg_block = 1 / (1 + np.exp(-0.062 * V) * self.mg_concentration / 3.57)
        return self.weight * self.g * mg_block * (V - 0)  # 0 mV is the reversal potential for NMDA receptors

    def update(self, t, dt):
        if self.pre_neuron.has_spiked(t):
            self.g += self.weight
        self.g += dt * (-self.g / self.tau_decay + (1 - self.g) / self.tau_rise)

class STDPSynapse(ExcitatorySynapse):
    def __init__(self, pre_neuron, post_neuron, weight=0.5, delay=1.0, tau=5.0, 
                 A_plus=0.005, A_minus=0.005, tau_plus=20.0, tau_minus=20.0):
        super().__init__(pre_neuron, post_neuron, weight, delay, tau)
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.pre_trace = 0.0
        self.post_trace = 0.0

    def update(self, t, dt):
        super().update(t, dt)
        
        # Update pre-synaptic trace
        if self.pre_neuron.has_spiked(t):
            self.pre_trace += 1
        self.pre_trace *= np.exp(-dt / self.tau_plus)
        
        # Update post-synaptic trace
        if self.post_neuron.has_spiked(t):
            self.post_trace += 1
        self.post_trace *= np.exp(-dt / self.tau_minus)
        
        # Update weight
        if self.pre_neuron.has_spiked(t):
            self.weight += self.A_plus * self.post_trace
        if self.post_neuron.has_spiked(t):
            self.weight -= self.A_minus * self.pre_trace
        
        # Ensure weight stays within bounds
        self.weight = np.clip(self.weight, 0, 1)

class SynapseFactory:
    @staticmethod
    def create_synapse(synapse_type, pre_neuron, post_neuron, **kwargs):
        if synapse_type == "excitatory":
            return ExcitatorySynapse(pre_neuron, post_neuron, **kwargs)
        elif synapse_type == "inhibitory":
            return InhibitorySynapse(pre_neuron, post_neuron, **kwargs)
        elif synapse_type == "nmda":
            return NMDASynapse(pre_neuron, post_neuron, **kwargs)
        elif synapse_type == "stdp":
            return STDPSynapse(pre_neuron, post_neuron, **kwargs)
        else:
            raise ValueError(f"Unknown synapse type: {synapse_type}")

class SynapticConnectionManager:
    def __init__(self):
        self.synapses = []

    def add_synapse(self, synapse):
        self.synapses.append(synapse)

    def remove_synapse(self, synapse):
        self.synapses.remove(synapse)

    def update_all_synapses(self, t, dt):
        for synapse in self.synapses:
            synapse.update(t, dt)

    def compute_total_psp(self, neuron, t):
        return sum(syn.compute_psp(t) for syn in self.synapses if syn.post_neuron == neuron)

class SynapticPlasticityManager:
    @staticmethod
    def apply_homeostatic_plasticity(neuron, target_activity, learning_rate):
        current_activity = neuron.get_average_activity()
        for synapse in neuron.incoming_synapses:
            synapse.weight += learning_rate * (target_activity - current_activity)
            synapse.weight = np.clip(synapse.weight, 0, 1)

    @staticmethod
    def apply_structural_plasticity(network, formation_probability, elimination_probability):
        for pre_neuron in network.neurons:
            for post_neuron in network.neurons:
                if pre_neuron != post_neuron:
                    if np.random.rand() < formation_probability:
                        new_synapse = SynapseFactory.create_synapse("excitatory", pre_neuron, post_neuron)
                        network.add_synapse(new_synapse)
                    
                    existing_synapses = [syn for syn in network.synapses if syn.pre_neuron == pre_neuron and syn.post_neuron == post_neuron]
                    for synapse in existing_synapses:
                        if np.random.rand() < elimination_probability:
                            network.remove_synapse(synapse)

# Example usage
if __name__ == "__main__":
    class MockNeuron:
        def __init__(self):
            self.V = -65
            self.spike_times = []
            self.incoming_synapses = []
        
        def has_spiked(self, t):
            return any(abs(t - spike_time) < 1e-6 for spike_time in self.spike_times)
        
        def get_average_activity(self):
            return len(self.spike_times) / 1000  # Assuming 1000 ms simulation
    
    pre_neuron = MockNeuron()
    post_neuron = MockNeuron()
    
    synapse_manager = SynapticConnectionManager()
    
    exc_synapse = SynapseFactory.create_synapse("excitatory", pre_neuron, post_neuron)
    inh_synapse = SynapseFactory.create_synapse("inhibitory", pre_neuron, post_neuron)
    nmda_synapse = SynapseFactory.create_synapse("nmda", pre_neuron, post_neuron)
    stdp_synapse = SynapseFactory.create_synapse("stdp", pre_neuron, post_neuron)
    
    # Add synapses to post_neuron's incoming_synapses
    post_neuron.incoming_synapses = [exc_synapse, inh_synapse, nmda_synapse, stdp_synapse]
    
    synapse_manager.add_synapse(exc_synapse)
    synapse_manager.add_synapse(inh_synapse)
    synapse_manager.add_synapse(nmda_synapse)
    synapse_manager.add_synapse(stdp_synapse)
    
    # Simulate for 1000 ms
    for t in range(1000):
        if t % 100 == 0:  # Spike every 100 ms
            pre_neuron.spike_times.append(t)
        
        synapse_manager.update_all_synapses(t, 1)
    
    print(f"Total PSP at t=150: {synapse_manager.compute_total_psp(post_neuron, 150)}")
    
    # Apply homeostatic plasticity
    SynapticPlasticityManager.apply_homeostatic_plasticity(post_neuron, target_activity=0.1, learning_rate=0.01)
    
    print(f"STDP synapse weight after simulation: {stdp_synapse.weight}")
    
    # Print weights of all synapses after homeostatic plasticity
    for i, synapse in enumerate(post_neuron.incoming_synapses):
        print(f"Synapse {i+1} weight after homeostatic plasticity: {synapse.weight}")
