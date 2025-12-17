"""Shapes:

NN: num_nets
BS: batch_size
ID: in_dim
OD: out_dim
LiID: layer_i_in_dim
LiOD: layer_i_out_dim
"""

import torch
from jaxtyping import Float
from torch import Tensor
from common.ne.pop._nets.static.base import BaseStaticNets, StaticNetsConfig

class FeedforwardStaticNets(BaseStaticNets):

    def __init__(self, config: StaticNetsConfig):
        self.config: StaticNetsConfig = config
        self.weights: list[Float[Tensor, "NN LiID LiOD"]] = []
        self.biases: list[Float[Tensor, "NN 1 LiOD"]] = []
        self.num_layers: int = len(self.config.layer_dims) - 1
        for i in range(self.num_layers):
            in_dim: int = self.config.layer_dims[i]
            out_dim: int = self.config.layer_dims[i + 1]
            std: float = (1.0 / in_dim) ** 0.5
            weight: Float[Tensor, "NN LiID LiOD"] = torch.randn(self.config.num_nets, in_dim, out_dim) * std
            bias: Float[Tensor, "NN 1 LiOD"] = torch.randn(self.config.num_nets, 1, out_dim) * std
            self.weights.append(weight)
            self.biases.append(bias)
        if self.config.sigma_sigma is not None:
            self.weight_sigmas: list[Float[Tensor, "NN LiID LiOD"]] = []
            self.bias_sigmas: list[Float[Tensor, "NN 1 LiOD"]] = []
            for weight, bias in zip(self.weights, self.biases):
                self.weight_sigmas.append(torch.full_like(weight, self.config.sigma))
                self.bias_sigmas.append(torch.full_like(bias, self.config.sigma))

    def __call__(self, x: Float[Tensor, "NN BS ID"]) -> Float[Tensor, "NN BS OD"]:
        for i in range(self.num_layers):
            x: Float[Tensor, "NN BS LiOD"] = torch.bmm(x, self.weights[i])
            x: Float[Tensor, "NN BS LiOD"] = x + self.biases[i]
            if i < self.num_layers - 1:
                x: Float[Tensor, "NN BS LiOD"] = torch.tanh(x)
        return x

    def mutate(self) -> None:
        for i in range(self.num_layers):
            if self.config.sigma_sigma is not None:
                xi: Float[Tensor, "NN LiID LiOD"] = torch.randn_like(self.weight_sigmas[i]) * self.config.sigma_sigma
                self.weight_sigmas[i]: Float[Tensor, "NN LiID LiOD"] = self.weight_sigmas[i] * (1 + xi)
                weight_sigma: Float[Tensor, "NN LiID LiOD"] = self.weight_sigmas[i]
                xi_bias: Float[Tensor, "NN 1 LiOD"] = torch.randn_like(self.bias_sigmas[i]) * self.config.sigma_sigma
                self.bias_sigmas[i]: Float[Tensor, "NN 1 LiOD"] = self.bias_sigmas[i] * (1 + xi_bias)
                bias_sigma: Float[Tensor, "NN 1 LiOD"] = self.bias_sigmas[i]
            else:
                weight_sigma: float = self.config.sigma
                bias_sigma: float = self.config.sigma
            self.weights[i]: Float[Tensor, "NN LiID LiOD"] = self.weights[i] + torch.randn_like(self.weights[i]) * weight_sigma
            self.biases[i]: Float[Tensor, "NN 1 LiOD"] = self.biases[i] + torch.randn_like(self.biases[i]) * bias_sigma
