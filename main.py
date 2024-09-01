import os

# Define the directory and file structure
structure = {
    "neuron_simulation": {
        "main.py": "",
        "neuron": {
            "__init__.py": "",
            "membrane.py": "# Neuron membrane properties",
            "ion_channels.py": "# Various ion channel models",
            "synapse.py": "# Synaptic transmission model",
            "action_potential.py": "# Action potential generation",
        },
        "utils": {
            "__init__.py": "",
            "numerical_methods.py": "# Numerical integration methods",
            "visualizations.py": "# Plotting and visualization functions",
        },
        "config": {
            "parameters.py": "# Simulation parameters",
        },
        "tests": {
            "test_membrane.py": "",
            "test_ion_channels.py": "",
            "test_synapse.py": "",
            "test_action_potential.py": "",
        },
        "data": {
            "simulation_results": {},  # Directory to store simulation outputs
        },
        "requirements.txt": "# Project dependencies",
        "README.md": "# Project documentation",
    }
}

def create_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            with open(path, 'w') as f:
                f.write(content)

# Set the base path to the current directory or any specific path you want
base_path = "."

# Create the directory structure
create_structure(base_path, structure)

print("Directory structure and files created successfully!")

