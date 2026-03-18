class MLPIntegratedGradientsHook:
    """在 down_proj 前按 alpha 缩放激活，并保存缩放后激活用于 IG 求导。"""

    def __init__(self):
        self.layer_activations = {}
        self.handles = []
        self.alpha = 1.0

    def set_alpha(self, alpha):
        self.alpha = alpha

    def register(self, model, layers):
        self.remove()
        for layer_idx in layers:
            mlp_module = model.model.layers[layer_idx].mlp

            def make_hook(captured_layer_idx):
                def hook_fn(module, inputs):
                    scaled_activation = inputs[0] * self.alpha
                    self.layer_activations[captured_layer_idx] = scaled_activation
                    return (scaled_activation,)

                return hook_fn

            handle = mlp_module.down_proj.register_forward_pre_hook(make_hook(layer_idx))
            self.handles.append(handle)

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.layer_activations = {}


class NeuronInterventionHook:
    """用于在生成过程中放大特定神经元激活值的 Hook。"""

    def __init__(self, target_neurons, multiplier=5.0):
        self.target_neurons = target_neurons
        self.multiplier = multiplier
        self.handles = []

    def register(self, model):
        for layer_idx, neuron_indices in self.target_neurons.items():
            mlp_module = model.model.layers[layer_idx].mlp

            def make_pre_hook(indices):
                def down_proj_pre_hook(module, inputs):
                    hidden_states = inputs[0].clone()
                    for n_idx in indices:
                        hidden_states[:, :, n_idx] = hidden_states[:, :, n_idx] * self.multiplier
                    return (hidden_states,)

                return down_proj_pre_hook

            handle = mlp_module.down_proj.register_forward_pre_hook(make_pre_hook(neuron_indices))
            self.handles.append(handle)

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
