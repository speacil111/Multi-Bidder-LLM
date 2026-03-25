import torch


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

                    # 仅干预最后一个 token，并约束该 token 向量 L2 norm 不变。
                    last_token = hidden_states[:, -1, :]
                    pre_norm = torch.linalg.vector_norm(last_token, ord=2, dim=-1, keepdim=True)

                    for n_idx in indices:
                        last_token[:, n_idx] = last_token[:, n_idx] * self.multiplier

                    post_norm = torch.linalg.vector_norm(last_token, ord=2, dim=-1, keepdim=True)
                    scale = torch.where(
                        post_norm > 0,
                        pre_norm / post_norm.clamp_min(1e-12),
                        torch.ones_like(post_norm),
                    )
                    last_token = last_token * scale
                    hidden_states[:, -1, :] = last_token
                    return (hidden_states,)

                return down_proj_pre_hook

            handle = mlp_module.down_proj.register_forward_pre_hook(make_pre_hook(neuron_indices))
            self.handles.append(handle)

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


class UnifiedInterventionHook:
    """多品牌统一干预：先把所有品牌的神经元各自放大，最后做一次整体 L2 norm 压缩。

    解决多个独立 Hook 各自 rescale 导致品牌间互相抵消的耦合问题。

    Parameters
    ----------
    concept_interventions : list of (neuron_map, multiplier)
        neuron_map = {layer_idx: [neuron_indices]}, multiplier = float
    """

    def __init__(self, concept_interventions):
        self.handles = []
        # 按 layer 聚合：{layer_idx: [(indices_tensor, multiplier), ...]}
        self._layer_groups = {}
        for neuron_map, multiplier in concept_interventions:
            for layer_idx, indices in neuron_map.items():
                if layer_idx not in self._layer_groups:
                    self._layer_groups[layer_idx] = []
                self._layer_groups[layer_idx].append((indices, multiplier))

    def register(self, model):
        self.remove()
        for layer_idx, groups in self._layer_groups.items():
            mlp_module = model.model.layers[layer_idx].mlp

            def make_pre_hook(layer_groups):
                def down_proj_pre_hook(module, inputs):
                    hidden_states = inputs[0].clone()
                    last_token = hidden_states[:, -1, :]
                    pre_norm = torch.linalg.vector_norm(
                        last_token, ord=2, dim=-1, keepdim=True
                    )

                    for indices, multiplier in layer_groups:
                        for n_idx in indices:
                            last_token[:, n_idx] = last_token[:, n_idx] * multiplier

                    post_norm = torch.linalg.vector_norm(
                        last_token, ord=2, dim=-1, keepdim=True
                    )
                    scale = torch.where(
                        post_norm > 0,
                        pre_norm / post_norm.clamp_min(1e-12),
                        torch.ones_like(post_norm),
                    )
                    last_token = last_token * scale
                    hidden_states[:, -1, :] = last_token
                    return (hidden_states,)

                return down_proj_pre_hook

            handle = mlp_module.down_proj.register_forward_pre_hook(
                make_pre_hook(groups)
            )
            self.handles.append(handle)

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
