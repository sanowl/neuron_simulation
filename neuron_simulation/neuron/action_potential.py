# Action potential generationimport numpy as np
from scipy.signal import find_peaks
import numpy as np

class ActionPotential:
    def __init__(self, neuron):
        self.neuron = neuron
        self.threshold = -55  # mV, typical threshold for action potential initiation
        self.refractory_period = 2  # ms
        self.last_spike_time = -np.inf

    def generate(self, T, dt, I_ext):
        """Generate an action potential with refractory period"""
        t, X = self.neuron.simulate(T, dt, I_ext)
        V = X[:, 0]  # Extract voltage
        
        # Find spike times
        spike_times = self.detect_spikes(t, V)
        
        # Apply refractory period
        valid_spikes = self.apply_refractory_period(spike_times)
        
        return t, V, valid_spikes

    def detect_spikes(self, t, V):
        """Detect spikes using a threshold crossing method"""
        spike_indices, _ = find_peaks(V, height=self.threshold, distance=int(self.refractory_period/t[1]))
        return t[spike_indices]

    def apply_refractory_period(self, spike_times):
        """Apply refractory period to spike times"""
        valid_spikes = []
        for spike_time in spike_times:
            if spike_time - self.last_spike_time > self.refractory_period:
                valid_spikes.append(spike_time)
                self.last_spike_time = spike_time
        return valid_spikes

    def analyze_waveform(self, t, V, spike_times):
        """Analyze action potential waveform"""
        results = []
        for spike_time in spike_times:
            # Find index of spike peak
            spike_index = np.argmin(np.abs(t - spike_time))
            
            # Extract waveform (10ms before and after spike)
            start = max(0, spike_index - int(10/t[1]))
            end = min(len(t), spike_index + int(10/t[1]))
            waveform = V[start:end]
            waveform_t = t[start:end] - t[start]
            
            # Calculate features
            amplitude = np.max(waveform) - V[start]
            width = self.calculate_width(waveform_t, waveform)
            rise_time = self.calculate_rise_time(waveform_t, waveform)
            decay_time = self.calculate_decay_time(waveform_t, waveform)
            
            results.append({
                'time': spike_time,
                'amplitude': amplitude,
                'width': width,
                'rise_time': rise_time,
                'decay_time': decay_time
            })
        
        return results

    def calculate_width(self, t, V):
        """Calculate action potential width at half maximum"""
        baseline = V[0]
        peak = np.max(V)
        half_amp = baseline + 0.5 * (peak - baseline)
        above_half = V >= half_amp
        rise_index = np.argmax(above_half)
        fall_index = len(V) - 1 - np.argmax(above_half[::-1])
        return t[fall_index] - t[rise_index]

    def calculate_rise_time(self, t, V):
        """Calculate rise time (10% to 90% of peak)"""
        baseline = V[0]
        peak = np.max(V)
        rise_10 = baseline + 0.1 * (peak - baseline)
        rise_90 = baseline + 0.9 * (peak - baseline)
        rise_10_index = np.argmax(V >= rise_10)
        rise_90_index = np.argmax(V >= rise_90)
        return t[rise_90_index] - t[rise_10_index]

    def calculate_decay_time(self, t, V):
        """Calculate decay time (90% to 10% of peak)"""
        baseline = V[0]
        peak = np.max(V)
        decay_90 = baseline + 0.9 * (peak - baseline)
        decay_10 = baseline + 0.1 * (peak - baseline)
        peak_index = np.argmax(V)
        decay_90_index = peak_index + np.argmax(V[peak_index:] <= decay_90)
        decay_10_index = peak_index + np.argmax(V[peak_index:] <= decay_10)
        return t[decay_10_index] - t[decay_90_index]
