import copy
import inspect
import itertools as I
import logging
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from transformers.models.mixtral.modeling_mixtral import (
    MixtralForCausalLM,
    MixtralSparseMoeBlock,
    MixtralBLockSparseTop2MLP,
)

from hcsmoe.data.dataset import CacheDataset
from hcsmoe.merging.grouping_mixtral import _merge_mlp_experts_by_usage_frequency_weighting, _merge_moe_experts_within_and_across_models

logger = logging.getLogger(__name__)

class PrunableMixtralSparseMoeBlockWrapper(torch.nn.Module):
    def __init__(self, model,
                 dom_experts: List[int],
                 r: Optional[int] = None,
                 usage_freq: Optional[torch.Tensor] = None,
                 merge_method: Optional[str] = "average",
                 mode: Optional[str] = "normal",
                 weight: Optional[List[int]] = None,
                 ):
        super().__init__()
        if isinstance(model, MixtralSparseMoeBlock):
            self.model = model
        else:
            self.model = model.model
        self.r = r

        self.experts_to_drop = None
        self.experts_assignment = None
        self.cache_space = CacheDataset()
        self.cache_logits = False
        self.cache_X = False # MoE input
        self.cache_Z = False # MoE output
        self.cache_R = False # MoE router selected experts
        self.dominant_experts = dom_experts
        self.usage_freq = usage_freq
        self.group_state_dict = {} # expert_idx: group_label
        # for i in range(self.r):
        #     self.group_state_dict[self.dominant_experts[i]] = i
        # self.normal_experts = [i for i in range(self.model.num_experts) if i not in self.dominant_experts]
        self.merge_method = merge_method
        self.mode = mode
        self.weight = weight
        self.device = self.model.gate.weight.device

    # Forward uses topk
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        # print model parameter device and hidden states device
        # print("model gate weight, gate weight shape {} on {}, hidden states shape {} on {}".format(
            # self.model.gate.weight.shape, self.model.gate.weight.device, hidden_states.shape, hidden_states.device))
        # hidden_states = hidden_states.to(self.model.gate.weight.device)
        router_logits = self.model.gate(hidden_states)

        if self.experts_to_drop is not None:
            for e in self.experts_to_drop:
                router_logits[:, e] = -float('inf')

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.model.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.model.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.model.num_experts):
            expert_layer = self.model.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None,
                                          top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(
                current_state * routing_weights[top_x_list, idx_list, None])

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype))

        if self.experts_to_drop is not None and (self.cache_logits or self.cache_X or self.cache_Z):
            logger.warn(
                f'Already dropped {self.experts_to_drop} but still storing activations.')
        self.cache_space.append(
            alpha=(router_logits if self.cache_logits else None),
            X=(hidden_states if self.cache_X else None),
            Z=(final_hidden_states if self.cache_Z else None),
            R=(selected_experts if self.cache_R else None),
        )
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim)

        return final_hidden_states, router_logits

    @torch.no_grad()
    def enumerate(self):
        self.cache_logits = False
        self.cache_X = False
        self.cache_Z = False
        self.cache_R = False
        loss_history = dict()
        
        self.cache_space.Rs = torch.concat(self.cache_space.Rs)
        cache_space_Xs = torch.concat(self.cache_space.Xs)

        with torch.inference_mode():
            for dropped in I.combinations(range(self.model.num_experts), self.model.num_experts - self.r):
                self.experts_to_drop = dropped
                loss = 0

                for (hidden_states, final_hidden_states) in zip(self.cache_space.Xs, self.cache_space.Zs):
                    hidden_states = hidden_states.to(
                        device=self.model.gate.weight.data.device, non_blocking=True)
                    final_hidden_states = final_hidden_states.to(
                        dtype=torch.float64, device=self.model.gate.weight.data.device, non_blocking=True)

                    final_hidden_states_e, _ = self.forward(
                        hidden_states.unsqueeze(0))
                    loss += torch.norm(final_hidden_states -
                                       final_hidden_states_e.squeeze(0).to(torch.float64)).item()
                loss_history[dropped] = loss
            self.experts_to_drop = None
        
       
        self.experts_to_drop = min(loss_history, key=loss_history.get)
        return loss_history
    
    @torch.no_grad()
    def merge(self):
        assert self.experts_assignment is not None
        assert len(self.experts_assignment) == self.model.num_experts - self.r
        self.cache_X = False
        self.cache_Z = False
        self.cache_R = False
        cache_space_Xs = torch.concat(self.cache_space.Xs)

        for i in range(self.model.num_experts - self.r):
            self.group_state_dict[self.normal_experts[i]] = self.experts_assignment[i]
        group_labels = [self.group_state_dict[key] for key in sorted(self.group_state_dict.keys())]
        print("merge: ", group_labels)

        if self.merge_method == "average":
            self.model = _merge_mlp_experts_by_usage_frequency_weighting(
                ffn=self.model,
                group_labels=torch.tensor(group_labels),
                usage_frequencies=torch.tensor([1] * self.model.num_experts),
            )
        elif self.merge_method == "weighted" and self.weight is not None:
            usage_frequencies = []
            for i in range(self.model.num_experts):
                if i in self.dominant_experts:
                    usage_frequencies.append(self.weight[0])
                else:
                    usage_frequencies.append(self.weight[1])
            self.model = _merge_mlp_experts_by_usage_frequency_weighting(
                ffn=self.model,
                group_labels=torch.tensor(group_labels),
                usage_frequencies=torch.tensor(usage_frequencies),
            )
        elif self.merge_method == "freq":
            self.model = _merge_mlp_experts_by_usage_frequency_weighting(
                ffn=self.model,
                group_labels=torch.tensor(group_labels),
                usage_frequencies=self.usage_freq,
            )
        elif self.merge_method == "zipit" or self.merge_method == "fix-dom-same":
            layer_forwarded_hidden_states = tuple()
            for expert_idx in range(self.model.num_experts):
                expert_mask = (self.cache_space.Rs == expert_idx)
                batch_tensor = torch.any(expert_mask, dim=-1)
                choice_input = cache_space_Xs[batch_tensor]
                layer_forwarded_hidden_states += (choice_input,)
            self.model = _merge_moe_experts_within_and_across_models(
                moe=self.model,
                group_labels=torch.tensor(group_labels),
                forwarded_hidden_states=layer_forwarded_hidden_states,
                dominant_alone=False,
                merge=self.merge_method,
                mode=self.mode,
                core_expert_indices=self.dominant_experts,
                usage_frequencies=None,
            )
            del layer_forwarded_hidden_states
        else:
            raise ValueError("Invalid merge method")

        print("merged_model: ")
        for e in range(self.model.num_experts):
            print(self.model.experts[e].w1.weight.data[0, :8])
        del self.cache_space
        print(torch.cuda.memory_summary())

    @torch.no_grad()
    def prune(self):
        assert self.experts_to_drop is not None
        assert len(self.experts_to_drop) == self.model.num_experts - self.r
        del self.cache_space
        self.cache_X = False
        self.cache_Z = False

        experts_to_reserve = sorted(
            set(range(self.model.num_experts)) - set(self.experts_to_drop))
        print("experts_to_reserve: ", experts_to_reserve)
        print("experts_to_drop: ", self.experts_to_drop)

        gate_new = torch.nn.Linear(in_features=self.model.gate.in_features,
                                   out_features=self.r, bias=False, device='cpu', dtype=torch.bfloat16)
        gate_new.weight.data = self.model.gate.weight.data[list(
            experts_to_reserve)]
