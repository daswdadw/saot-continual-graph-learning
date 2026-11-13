"""GCN/GAT encoders used by the member 3 replay and SAOT modules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch_geometric.nn import GATConv, GCNConv

EncoderType = Literal["gcn", "gat"]


def _validate_edge_index(edge_index: Tensor, num_nodes: int) -> None:
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(
            "edge_index must have shape [2, num_edges], "
            f"got {tuple(edge_index.shape)}."
        )
    if edge_index.dtype not in (torch.int32, torch.int64):
        raise TypeError("edge_index must contain integer node indices.")
    if edge_index.numel() == 0:
        return
    minimum = int(edge_index.min())
    maximum = int(edge_index.max())
    if minimum < 0 or maximum >= num_nodes:
        raise ValueError(
            "edge_index contains an out-of-range node index: "
            f"valid range is [0, {num_nodes - 1}], found [{minimum}, {maximum}]."
        )


class GNNEncoder(nn.Module):
    """Configurable node encoder with GCN and GAT backends.

    The default is the required two-layer GCN.  For Reddit-CL, use
    ``encoder_type='gat'`` and ``num_layers=3``.  Hidden GAT layers use
    ``gat_heads`` heads with ``concat=False`` so that the representation width
    remains ``hidden_dim`` and the next layer has a stable interface.

    Args:
        input_dim: Number of input node features.
        hidden_dim: Width of hidden representations.
        output_dim: Final representation width; defaults to ``hidden_dim``.
        encoder_type: ``'gcn'`` or ``'gat'``.
        num_layers: Number of graph convolution layers, at least one.
        dropout: Dropout applied between layers and after intermediate activations.
        gat_heads: Number of attention heads in hidden GAT layers.
        add_self_loops: Whether convolution layers add self loops.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int | None = None,
        *,
        encoder_type: EncoderType = "gcn",
        num_layers: int = 2,
        dropout: float = 0.0,
        gat_heads: int = 4,
        add_self_loops: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError(f"input_dim must be a positive integer, got {input_dim!r}.")
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be a positive integer, got {hidden_dim!r}.")
        if output_dim is None:
            output_dim = hidden_dim
        if isinstance(output_dim, bool) or not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(f"output_dim must be a positive integer, got {output_dim!r}.")
        if encoder_type not in ("gcn", "gat"):
            raise ValueError(f"encoder_type must be 'gcn' or 'gat', got {encoder_type!r}.")
        if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError(f"num_layers must be a positive integer, got {num_layers!r}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        if isinstance(gat_heads, bool) or not isinstance(gat_heads, int) or gat_heads < 1:
            raise ValueError(f"gat_heads must be a positive integer, got {gat_heads!r}.")
        if not isinstance(add_self_loops, bool):
            raise TypeError("add_self_loops must be a boolean.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.encoder_type = encoder_type
        self.num_layers = num_layers
        self.dropout = float(dropout)
        self.gat_heads = gat_heads

        dimensions = [input_dim]
        if num_layers > 1:
            dimensions.extend([hidden_dim] * (num_layers - 1))
        dimensions.append(output_dim)

        layers: list[nn.Module] = []
        for layer_index in range(num_layers):
            in_channels = dimensions[layer_index]
            out_channels = dimensions[layer_index + 1]
            if encoder_type == "gcn":
                layer = GCNConv(
                    in_channels,
                    out_channels,
                    add_self_loops=add_self_loops,
                    normalize=True,
                )
            else:
                heads = gat_heads if layer_index < num_layers - 1 else 1
                layer = GATConv(
                    in_channels,
                    out_channels,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                    add_self_loops=add_self_loops,
                )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self.activation = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset convolution parameters using PyG's standard initialisers."""

        for layer in self.layers:
            layer.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        *,
        edge_weight: Tensor | None = None,
        return_all_layers: bool = False,
    ) -> Tensor | tuple[Tensor, ...]:
        """Encode node features on a local or full graph.

        ``edge_index`` may use local node IDs, as returned by
        :func:`data.neighbor_sampling.sample_layerwise`.  When ``return_all_layers``
        is true, a tuple containing the output after each graph convolution is
        returned; otherwise only the final representation is returned.
        """

        if x.ndim != 2:
            raise ValueError(f"x must have shape [num_nodes, num_features], got {tuple(x.shape)}.")
        if x.size(1) != self.input_dim:
            raise ValueError(
                f"x has {x.size(1)} features, but encoder expects {self.input_dim}."
            )
        if x.numel() == 0:
            raise ValueError("x must contain at least one node.")
        if x.device != edge_index.device:
            raise ValueError("x and edge_index must be on the same device.")
        _validate_edge_index(edge_index, x.size(0))
        if edge_weight is not None:
            if edge_weight.ndim != 1 or edge_weight.numel() != edge_index.size(1):
                raise ValueError(
                    "edge_weight must have shape [num_edges], "
                    f"got {tuple(edge_weight.shape)} for {edge_index.size(1)} edges."
                )
            if edge_weight.device != edge_index.device:
                raise ValueError("edge_weight and edge_index must be on the same device.")
        if not isinstance(return_all_layers, bool):
            raise TypeError("return_all_layers must be a boolean.")

        representations: list[Tensor] = []
        hidden = x
        for layer_index, layer in enumerate(self.layers):
            if self.encoder_type == "gcn":
                hidden = layer(hidden, edge_index, edge_weight=edge_weight)
            else:
                # GATConv has no scalar edge_weight argument. Edge attributes
                # can be added later without changing the public interface.
                hidden = layer(hidden, edge_index)
            if layer_index < self.num_layers - 1:
                hidden = self.activation(hidden)
                hidden = nn.functional.dropout(
                    hidden, p=self.dropout, training=self.training
                )
            representations.append(hidden)

        return tuple(representations) if return_all_layers else hidden

    def encode(self, x: Tensor, edge_index: Tensor, **kwargs: Any) -> Tensor | tuple[Tensor, ...]:
        """Alias for ``forward`` used by training code."""

        return self.forward(x, edge_index, **kwargs)


def build_gnn_encoder(
    config: Mapping[str, Any],
    *,
    input_dim: int,
    output_dim: int | None = None,
) -> GNNEncoder:
    """Build an encoder from a YAML-like configuration mapping."""

    encoder_type = str(config.get("encoder_type", "gcn")).lower()
    num_layers = int(config.get("num_layers", 2 if encoder_type == "gcn" else 3))
    return GNNEncoder(
        input_dim=input_dim,
        hidden_dim=int(config.get("hidden_dim", 256)),
        output_dim=(
            output_dim
            if output_dim is not None
            else config.get("output_dim")
        ),
        encoder_type=encoder_type,  # type: ignore[arg-type]
        num_layers=num_layers,
        dropout=float(config.get("dropout", 0.0)),
        gat_heads=int(config.get("gat_heads", 4)),
        add_self_loops=bool(config.get("add_self_loops", True)),
    )
