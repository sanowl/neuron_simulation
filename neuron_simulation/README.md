# Neuron Membrane Simulation

This project implements a detailed simulation of a neuron's membrane dynamics, including ion channels, synapses, and basic plasticity mechanisms.

## Features

- Morphology-based neuron model with compartments for soma, dendrites, and axon
- Ion channel models: Sodium, Potassium, and Leak channels
- Basic synapse model with conductance-based current
- Hodgkin-Huxley style differential equations for membrane potential
- Spike-Timing Dependent Plasticity (STDP) implementation
- Simulation using SciPy's ODE solver

## Requirements

- Python 3.7+
- NumPy
- SciPy
- Matplotlib (for visualization)
- Numba (for performance optimization)

## Usage

1. Ensure you have all required libraries installed:
   ```
   pip install numpy scipy matplotlib numba
   ```

2. Prepare a neuron morphology XML file (e.g., `neuron_morphology.xml`).

3. Run the simulation:
   ```python
   from membrane import NeuronMembrane

   # Initialize the neuron
   neuron = NeuronMembrane("neuron_morphology.xml")

   # Define external current function
   def I_ext_func(t):
       return np.array([10 if 100 <= t <= 300 else 0 for _ in range(len(neuron.compartments))])

   # Run simulation
   t, result = neuron.simulate(500, 0.1, I_ext_func)

   # Plot results
   import matplotlib.pyplot as plt
   plt.plot(t, result['V'][0])
   plt.xlabel('Time (ms)')
   plt.ylabel('Membrane Potential (mV)')
   plt.title('Soma Membrane Potential')
   plt.show()
   ```

## Extending the Model

You can extend this model by:
- Adding more ion channel types
- Implementing more complex synaptic models
- Enhancing the plasticity mechanisms
- Adding more detailed calcium dynamics

## License

This project is open-source and available under the MIT License.
