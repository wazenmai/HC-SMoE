from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from transformers.activations import ACT2FN

def merged_qwen2moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """ """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    # One hot encode the selected experts to create an expert mask
    # this will be used to easily index which expert is going to be sollicitated
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

    # Loop over all available experts in the model and perform the computation on each expert
    for expert_idx in range(self.num_experts):
        expert_layer = self.experts[self.expert_dict[expert_idx]]
        idx, top_x = torch.where(expert_mask[expert_idx])

        # Index the correct hidden states and compute the expert hidden state for
        # the current expert. We need to make sure to multiply the output hidden
        # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

        # However `index_add_` only support torch tensors for indexing so we'll use
        # the `top_x` tensor here.
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

    shared_expert_output = self.shared_expert(hidden_states)
    shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

    final_hidden_states = final_hidden_states + shared_expert_output

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits

class Qwen2MoEWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.expert_to_group = {} # expert_idx: group_label
        self.group_to_expert = {} # group label: [expert idx]
        self.unmerge_matrix = {} # group label: unmerge matrix for down_proj
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.model.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.model.top_k, dim=-1)
        if self.model.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.model.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.model.num_experts):
            expert_layer = self.model.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            group_label = self.expert_to_group[expert_idx]
            if len(self.group_to_expert[group_label]) == 1:
                group_idx = 0
            else:
                group_idx = torch.where(self.group_to_expert[group_label] == expert_idx)[0].item()
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            if self.unmerge_matrix[group_label] is not None:
                current_hidden_states = torch.matmul(expert_layer(current_state), self.unmerge_matrix[group_label][:, group_idx * hidden_dim:(group_idx+1) * hidden_dim]) * routing_weights[top_x, idx, None]
            else:
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        shared_expert_output = self.model.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(self.model.shared_expert_gate(hidden_states)) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class SharedLinearLayers(nn.Module):
    def __init__(self, config, shared_gate_proj, shared_down_proj, shared_up_proj):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        
        self.gate_proj_layers = shared_gate_proj
        self.down_proj_layers = shared_down_proj
        self.up_proj_layers = shared_up_proj

class ModifiedQwenBlockSparseTop2MLP(nn.Module):
    def __init__(self, config, shared_layers, gate_proj_id, down_proj_id, up_proj_id):
        super().__init__()
        self.shared_layers = shared_layers
        self.act_fn = ACT2FN[config.hidden_act]
        self.gate_proj_id = gate_proj_id
        self.down_proj_id = down_proj_id
        self.up_proj_id = up_proj_id

    def forward(self, hidden_states):
        gate_proj = self.shared_layers.gate_proj_layers[self.gate_proj_id]
        down_proj = self.shared_layers.down_proj_layers[self.down_proj_id]
        up_proj = self.shared_layers.up_proj_layers[self.up_proj_id]
        
        current_hidden_states = self.act_fn(gate_proj(hidden_states)) * up_proj(hidden_states)
        current_hidden_states = down_proj(current_hidden_states)
        return current_hidden_states


class ModifiedQwen2MoeSparseMoeBlock(nn.Module):
    def __init__(self, config, module_dict, shared_gate_proj, shared_down_proj, shared_up_proj, device, dtype):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts  # Fixed to 4 experts as per the requirement
        self.top_k = config.num_experts_per_tok

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False, device=device, dtype=dtype)

        # Shared linear layers
        self.shared_layers = SharedLinearLayers(config, shared_gate_proj, shared_down_proj, shared_up_proj)

        # Experts
        self.experts = nn.ModuleList([
            ModifiedQwenBlockSparseTop2MLP(config, self.shared_layers, module_dict["gate_proj"][i], module_dict["down_proj"][i], module_dict["up_proj"][i])
            for i in range(self.num_experts)
        ])

        # Jitter parameters
        self.jitter_noise = config.router_jitter_noise

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits



