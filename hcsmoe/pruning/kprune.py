import torch
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from tqdm import tqdm

from transformers import MixtralForCausalLM, MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock, MixtralBlockSparseTop2MLP

import time

class KPruner(object):
    def __init__(
            self,
            config,
            start_layer,
            reconstruct_batch_size,
            lam_pred,   # coef for predictive knowledge
            lam_rep,    # coef for representative knowledge
            mu,         # coef for attention heads score
            T,          # temperature for softmax
            constraint, # allowed budget, 1.0 means no pruning
            reconstruct=False,
    ):
        self.reconstruct_batch_size = reconstruct_batch_size
        self.lam_pred = lam_pred
        self.lam_rep = lam_rep
        self.mu = mu
        self.T = T
        self.constraint = constraint
        # self.max_params = model.num_parameters() * constraint
        self.reconstruct = reconstruct

        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.num_experts = config.num_local_experts # for mixtral
        self.d_model = config.hidden_size
        self.d_head = self.d_model // self.num_heads
        self.d_ff = config.intermediate_size
        self.topk = config.num_experts_per_tok # for mixtral
        self.prune_layer_indices = [i for i in range(start_layer, self.num_layers)]

        self.f_moe = self.d_ff * self.d_model * 3 * self.num_experts
        self.f_expert = self.d_ff * self.d_model * 3
        self.f_neuron = self.d_model * 3 * self.num_experts # if we prune neurons in every experts in that layer
        self.f_head = self.d_head * self.d_model * 3 * self.num_experts

    def kprune_for_mixtral_layerwise(
            self,
            model: MixtralForCausalLM,
            dataloader: DataLoader, 
            kd_labels,
            kd_outputs,
            layer_idx,
    ):
        # TODO: prune attention heads
        # NOW: prune block_sparse_moe only

        _st = time.time()
        
        # Initialization
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # STEP: 1. compute knowledge
        # 1.1 Initialization
        moe = model.model.layers[layer_idx].block_sparse_moe
        experts = model.model.layers[layer_idx].block_sparse_moe.experts
        
        _device = experts[0].w2.weight.device
        moe_rep_kl = torch.zeros(self.num_experts, self.d_ff).to(_device)
        moe_pred_kl = torch.zeros(self.num_experts, self.d_ff).to(_device)
        moe_masks = torch.ones(self.num_experts, self.d_ff, dtype=torch.float16).to(_device)
        moe_masks.requires_grad_(True)

        handles = []
        _inputs = {} # input fo single expert- compute score
        # layer_inputs = [] # input for layer - reconstruction
        expert_activations = {} # input for expert's output_proj  - reconstruction
        router_logits_rc = [] # router logits of moe layer - reconstruction
        expert_index_rc = [] # expert index of moe layer - reconstruction

        def apply_mask(module, _mask):
            # applying masks to the input to compute gradients
            def masking(_, i):
                return _mask * i[0]

            handle = module.register_forward_pre_hook(masking)
            return handle
        
        def hijack(module, _list, _hijack_input, _stop_forward=False):
            # if _stop_forward=True, then it raise error after forwarding the module
            if _hijack_input:
                def input_hook(_, inputs, __):
                    _list.append(inputs[0].detach().cpu()) # .clone().data
                    if _stop_forward:
                        raise StopFowardException

                handle = module.register_forward_hook(input_hook)
            else:
                def output_hook(_, __, outputs):
                    if isinstance(outputs, tuple):
                        _list.append(outputs[0].detach().cpu())
                    else:
                        _list.append(outputs.detach()) # .clone().data
                    if _stop_forward:
                        raise StopFowardException

                handle = module.register_forward_hook(output_hook)
            return handle

        # 1.2 Register hook function
        
        # handles.append(
        #     hijack(moe, layer_inputs, _hijack_input=True)
        # )
        for e in range(self.num_experts):
            # Apply mask - expert
            handles.append(
                apply_mask(experts[e].w2, moe_masks[e])
            )
            # Apply input hook - expert
            _inputs[e] = []
            handles.append(
                hijack(experts[e].w2, _inputs[e], _hijack_input=True)
            )
            expert_activations[e] = []
        if self.reconstruct and layer_idx == self.prune_layer_indices[0]:
            for sl in self.prune_layer_indices:
                kd_outputs[sl] = []
                handles.append(
                    hijack(model.model.layers[sl].block_sparse_moe, kd_outputs[sl], _hijack_input=False)
                )

        # 1.3 Forward pass
        num_tokens = 0
        num_samples = 0
        _index = 0
        for b, batch in enumerate(dataloader):
            print(b, end='')
            batch = {k: v.cuda() for k, v in batch.items()}
            att_mask = batch['attention_mask'].bool() # (batch_size, d_model)
            num_tokens += batch['attention_mask'].sum()
            batch_samples = batch['attention_mask'].shape[0]
            num_samples += batch_samples

            outputs = model(**batch, output_router_logits=True)

            if layer_idx == self.prune_layer_indices[0]:
                pred = F.softmax(outputs.logits / self.T, dim=1).detach()
                kd_labels.append(pred.cpu())
            else:
                pred = kd_labels[b].to(outputs.logits.device)
            kl_div = F.kl_div(
                input=F.log_softmax(outputs.logits / self.T, dim=1),
                target=pred,
                reduction="batchmean",
            ) * (self.T ** 2)
            
            kl_div /= 100
            print("kl_div: ", kl_div.item())
            kl_div.backward()

            
            router_logits = outputs.router_logits # a tuple, length: num_layers, each element: (batch_size * d_model, num_expert)

            # global collect all unpruned layers core to get gloabl mask
            routing_weights = F.softmax(router_logits[layer_idx], dim=-1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, self.topk, dim=-1)
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            if self.reconstruct:
                router_logits_rc.append(routing_weights.to(_device))
                expert_index_rc.append(selected_experts.to(_device))
            for e in range(self.num_experts):
                _weight = model.model.layers[layer_idx].block_sparse_moe.experts[e].w2.weight
                token_id = (selected_experts == e).nonzero()
                number_of_tokens = token_id.shape[0]
                _features = _inputs[e][-1][:number_of_tokens].to(torch.float32).to(_weight.device)
                print(f"_features: {torch.max(_features)}")
                if self.reconstruct:
                    expert_activations[e].append(_features)
                moe_rep_kl[e] += ((_features ** 2).sum(dim=0) * (_weight ** 2).mean(dim=0)).data

                grad = moe_masks.grad[e]
                moe_pred_kl[e] += (grad.detach() ** 2) * 0.5
                del _inputs[e][-1]
            moe_masks.grad = None

        # 1.4 Averaging score
        moe_rep_kl /= num_samples
        moe_pred_kl /= num_tokens.to(_device)

        print(f"moe_rep_kl: {moe_rep_kl.shape} {moe_rep_kl}")
        print(f"moe_pred_kl: {moe_pred_kl.shape} {moe_pred_kl}")

        # 1.5 Compute score
        moe_scores = (self.lam_rep * moe_rep_kl + self.lam_pred * moe_pred_kl)
        print(f"moe_scores: {moe_scores.shape} {moe_scores}")
        if layer_idx == self.prune_layer_indices[0]:
            kd_labels = torch.cat(kd_labels, dim=0)
            print("kd_labels: ", kd_labels.shape)
            # for sl in range(self.num_layers):
                # kd_outputs[sl] = torch.cat(kd_outputs[sl], dim=0)
                # print(f"kd_outputs[{sl}]: {kd_outputs[sl].shape}")
        # layer_inputs = torch.cat(layer_inputs, dim=0)
        # print(f"layer_inputs: {layer_inputs.shape}")
        # for e in range(self.num_experts):
            # expert_activations[e] = torch.cat(expert_activations[e], dim=0)
            # print(f"expert_activations[{e}]: {expert_activations[e].shape}")

        for handle in handles:
            handle.remove()
        
        del _inputs, handles


        # STEP: 2. mask search
        moe_scores_average = moe_scores.mean(dim=0) # (E, N) -> (N)
        s_tilde = moe_scores_average.sort().values
        print("s_tilde: ", s_tilde)
        threshold_pos = int(s_tilde.shape[0] * (1 - self.constraint))
        print(f"threshold_pos: {threshold_pos}, threshold: {s_tilde[threshold_pos]}")
        pruning_mask = (moe_scores_average > s_tilde[threshold_pos])
        left_neurons = pruning_mask.sum().item()
        print(f"pruning_mask: {pruning_mask.shape} left neurons: {left_neurons}")

        # Prune the layer
        for e in range(self.num_experts):
            print(f"Expert {e}: {torch.max(experts[e].w2.weight.data)} {torch.min(experts[e].w2.weight.data)} {torch.mean(experts[e].w2.weight.data)}")
            experts[e].w1 = prune_linear_layer(experts[e].w1, pruning_mask, dim=0)
            experts[e].w2 = prune_linear_layer(experts[e].w2, pruning_mask, dim=1)
            experts[e].w3 = prune_linear_layer(experts[e].w3, pruning_mask, dim=0)
       

        # STEP: 3. reconstruct pruned layer
        # Need
            # Student
                # layer_inputs:          Input of moe layer (B, T, d_model)
                # expert_activations:    Input of moe layer' expert e's output_proj (E, B, Ｔ, d_ff)
                # router_logits_rc:      Router logits of moe layer (B, T, 2)
                # expert_index_rc:       Expert index of moe layer (B, T, 2)
            # Teacher
                # kd_outputs[layer_idx]: Output of moe layer (B, T, d_model)
        
        # print(f"layer_inputs: {len(layer_inputs)} {layer_inputs[0].shape}")
        if self.reconstruct:
            print(f"expert_activations: ({len(expert_activations)}, {len(expert_activations[e])}, {expert_activations[0][0].shape}, {expert_activations[1][0].shape})")
            print(f"router_logits_rc: ({len(router_logits_rc)}, {router_logits_rc[0].shape})")
            print(f"expert_index_rc: {len(expert_index_rc)} {len(expert_index_rc[0])} {expert_index_rc[0][0].shape}")
            print(f"kd_outputs[layer_idx]: {len(kd_outputs[layer_idx])} {kd_outputs[layer_idx][0].shape}")

            print(f"({layer_idx:2d}) [MoE] {(~pruning_mask).sum().item()} neurons pruned")

            num_batches = len(router_logits_rc)
            target_device = experts[e].w2.weight.device
            compute_device = torch.device("cuda:0") # torch.device("cuda:7")
            mse = torch.nn.MSELoss()

            expert_input = []
            for b in range(num_batches):
                num_tokens = router_logits_rc[b].shape[0]
                batch_input = torch.zeros(num_tokens, self.num_experts * left_neurons, dtype=torch.float, device=compute_device)
                token_index_for_each_experts = [0 for _ in range(self.num_experts)]
                for t in range(num_tokens):
                    first_expert = expert_index_rc[b][t][0].item()
                    second_expert = expert_index_rc[b][t][1].item()
                    batch_input[t][first_expert * left_neurons : (first_expert + 1) * left_neurons] = expert_activations[first_expert][b][token_index_for_each_experts[first_expert]][pruning_mask] * router_logits_rc[b][t][0]
                    batch_input[t][second_expert * left_neurons : (second_expert + 1) * left_neurons] = expert_activations[second_expert][b][token_index_for_each_experts[second_expert]][pruning_mask] * router_logits_rc[b][t][1]
                    token_index_for_each_experts[first_expert] += 1
                    token_index_for_each_experts[second_expert] += 1
                expert_input.append(batch_input)
            expert_input = torch.cat(expert_input, dim=0).to(torch.float)
            del expert_activations, router_logits_rc, expert_index_rc

            for dim in range(0, self.d_model, self.reconstruct_batch_size):
                print(f"Dimension {dim} / {self.d_model}")
                expert_output = torch.cat([
                    kd_outputs[layer_idx][b][:, :, dim:dim + self.reconstruct_batch_size].reshape(-1, self.reconstruct_batch_size)
                    for b in range(num_batches)
                ], dim=0).to(torch.float).to(compute_device)
                res = torch.linalg.lstsq(expert_input, expert_output, rcond=0.)
                W_new = res.solution.T
                print(f"W_new: {W_new.shape} {torch.isnan(W_new).sum()} {torch.isinf(W_new).sum()} {torch.max(W_new)} {torch.min(W_new)} {torch.mean(W_new)}")
                residuals = mse(expert_input @ W_new.T, expert_output)
                print(f"residuals: {residuals}")
                W_new.to(target_device)
                del expert_output
                for e in range(self.num_experts):
                    # d_model * d_ff -> d_model * left_neurons
                    experts[e].w2.weight[dim:dim + self.reconstruct_batch_size] = W_new[:, e * left_neurons : (e + 1) * left_neurons]
            del expert_input
            torch.cuda.empty_cache()
            kd_outputs[layer_idx].clear()
            del kd_outputs[layer_idx]

        # for dim in range(0, self.d_model, self.reconstruct_batch_size):
        #     expert_input = []
        #     expert_output = []
        #     print(f"Dimension {dim} / {self.d_model}")
            

        #     for b in range(num_batches):
        #         # print("Batch ", b)
        #         num_tokens = router_logits_rc[b].shape[0]
        #         batch_input = torch.zeros(num_tokens, self.num_experts * left_neurons, dtype=torch.float, device=target_device) # (T, E x d_ff)
        #         batch_output = kd_outputs[layer_idx][b][:, dim:dim + self.reconstruct_batch_size].reshape(num_tokens, self.reconstruct_batch_size)
        #         # print(f"batch_input: {batch_input.shape}")

        #         num_batches = len(router_logits_rc)
        
        #         # Create indices for the first and second experts
        #         first_expert_indices = expert_index_rc[b][:, 0] # (T)
        #         second_expert_indices = expert_index_rc[b][:, 1] # (T)
                
        #         # Compute cumulative sum of tokens for each expert
        #         expert_cumsum = torch.zeros(self.num_experts + 1, dtype=torch.long, device=target_device)
        #         for i in range(self.num_experts):
        #             expert_cumsum[i + 1] = expert_cumsum[i] + (first_expert_indices == i).sum() + (second_expert_indices == i).sum()
                
        #         # Compute token indices for each expert
        #         first_token_indices = torch.zeros_like(first_expert_indices)
        #         second_token_indices = torch.zeros_like(second_expert_indices)
        #         for i in range(self.num_experts):
        #             first_mask = (first_expert_indices == i)
        #             second_mask = (second_expert_indices == i)
        #             mask = torch.logical_or(first_mask, second_mask)
        #             index = mask.nonzero()
        #             first_token_indices[mask] = torch.arange(mask.sum(), device=target_device) + (expert_cumsum[i] - mask.sum())
        #             mask = (second_expert_indices == i)
        #             second_token_indices[mask] = torch.arange(mask.sum(), device=target_device) + expert_cumsum[i]
                
        #         # Gather the expert activations
        #         first_activations = expert_activations[first_expert_indices][b, first_token_indices][:, pruning_mask]
        #         second_activations = expert_activations[second_expert_indices][b, second_token_indices][:, pruning_mask]
                
        #         # Compute offsets for each expert
        #         first_expert_offsets = first_expert_indices * left_neurons
        #         second_expert_offsets = second_expert_indices * left_neurons
                
        #         # Compute the indices for the input tensor
        #         first_indices = first_expert_offsets[:, None] + torch.arange(left_neurons, device=target_device)
        #         second_indices = second_expert_offsets[:, None] + torch.arange(left_neurons, device=target_device)
                
        #         # Multiply by router logits and assign to batch_input
        #         batch_input.scatter_(1, first_indices, first_activations * router_logits_rc[b][:, 0:1])
        #         batch_input.scatter_(1, second_indices, second_activations * router_logits_rc[b][:, 1:2])
        


        #         token_index_for_each_experts = [0 for _ in range(self.num_experts)]
        #         for t in range(num_tokens):
        #             # print("Token ", t)
        #             first_expert = expert_index_rc[b][t][0].item()
        #             # print(f"first_expert: {first_expert} {router_logits_rc[b][t][0]}, token_idx: {token_index_for_each_experts[first_expert]} activation:  {expert_activations[first_expert][b].shape}")
        #             batch_input[t][first_expert * left_neurons : (first_expert + 1) * left_neurons] = expert_activations[first_expert][b][token_index_for_each_experts[first_expert]][pruning_mask] * router_logits_rc[b][t][0]
        #             token_index_for_each_experts[first_expert] += 1

        #             second_expert = expert_index_rc[b][t][1].item()
        #             # print(f"second_expert: {second_expert} {router_logits_rc[b][t][1]}, token_idx: {token_index_for_each_experts[second_expert]} activation:  {expert_activations[second_expert][b].shape}")
        #             batch_input[t][second_expert * left_neurons : (second_expert + 1) * left_neurons] = expert_activations[second_expert][b][token_index_for_each_experts[second_expert]][pruning_mask] * router_logits_rc[b][t][1]
        #             token_index_for_each_experts[second_expert] += 1
        #         expert_input.append(batch_input)
        #         expert_output.append(batch_output) # T x 1
        #     expert_input = torch.cat(expert_input, dim=0).to(torch.float).to(target_device) # (B x T, E x d_ff)
        #     expert_output = torch.cat(expert_output, dim=0).to(torch.float).to(target_device) # (B x T, 1)
        #     # print(f"expert_input: {expert_input.shape} {expert_input.dtype} {expert_input.device} {torch.max(expert_input)} {torch.min(expert_input)} {torch.mean(expert_input)} {expert_input[0]}")
        #     # print(f"expert_output: {expert_output.shape} {expert_output.dtype} {expert_output.device} {torch.max(expert_output)} {torch.min(expert_output)} {torch.mean(expert_output)} {expert_output[0]}")
        #     res = torch.linalg.lstsq(expert_input, expert_output, rcond=0.)
        #     W_new = res.solution.to(target_device).T
        #     print(f"W_new: {W_new.shape} {torch.isnan(W_new).sum()} {torch.isinf(W_new).sum()} {torch.max(W_new)} {torch.min(W_new)} {torch.mean(W_new)}")
        #     residuals = mse(expert_input @ W_new.T, expert_output)
        #     print(f"residuals: {residuals}")

        #     for e in range(self.num_experts):
        #         # d_model * d_ff -> d_model * left_neurons
        #         experts[e].w2.weight[dim:dim + self.reconstruct_batch_size] = W_new[:, e * left_neurons : (e + 1) * left_neurons]
        
        print(f"[Pruning] Time: {time.time() - _st:.2f}s")
        # print(torch.cuda.memory_summary())

    
    def kprune_for_mixtral(
            self,
            model,
            dataloader, 
    ):
        kd_labels = []
        kd_outputs = {}
        _all_st = time.time()
        for layer_idx in self.prune_layer_indices:
            print(f"[Pruning] ----------Target: layer {layer_idx}----------")
            self.kprune_for_mixtral_layerwise(
                model=model,
                dataloader=dataloader,
                kd_labels=kd_labels,
                kd_outputs=kd_outputs,
                layer_idx=layer_idx,
            )
        print(f"[Pruning] Total time: {time.time() - _all_st:.2f}s")
        return model

def remove_paddings(_value, att_mask):
    if len(_value.shape) == 2:
        return _value[att_mask]
    return _value[att_mask, :]

def prune_linear_layer(layer, _mask, dim):
    _device = layer.weight.device
    if dim == 1:
        W = layer.weight[:, _mask].detach()
    else:
        W = layer.weight[_mask, :].detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.detach()
        else:
            b = layer.bias[surv_mask].detach()
    W.to(torch.float16)
    new_size = list(layer.weight.size())
    new_size[dim] = torch.sum(_mask).item()
    new_layer = torch.nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None, device=_device, dtype=torch.float16)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W)
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b)
    return new_layer

def update_output_proj(_module, new_weight):
    _module.weight.requires_grad = False
    _module.weight.copy_(new_weight)