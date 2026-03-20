# Reward Model Worker Code Trace

This document traces the reward model worker initialization and usage flow in VERL.

## File Call Chain

### 1. **Initialization Flow** (Model Loading)

```
A. verl/trainer/main_ppo.py (lines 149-156)
   └─> Determines which RewardModelWorker class to use based on strategy
       ├─> If strategy == "fsdp" or "fsdp2":
       │   └─> from verl.workers.fsdp_workers import RewardModelWorker
       └─> If strategy == "megatron":
           └─> from verl.workers.megatron_workers import RewardModelWorker
   
   └─> Creates role_worker_mapping:
       role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)

B. verl/trainer/ppo/ray_trainer.py (lines 750-757)
   └─> RayPPOTrainer.__init__() receives role_worker_mapping
   └─> Creates RayClassWithInitArgs wrapper:
       rm_cls = RayClassWithInitArgs(
           self.role_worker_mapping[Role.RewardModel], 
           config=self.config.reward_model
       )
   └─> Stores in resource_pool_to_cls[resource_pool]["rm"] = rm_cls

C. verl/trainer/ppo/ray_trainer.py (lines 768-784)
   └─> init_workers() method:
       ├─> Creates WorkerGroup via ray_worker_group_cls.spawn()
       └─> Calls: self.rm_wg.init_model()  (line 784)

D. verl/single_controller/ray/base.py (WorkerGroup.init_model)
   └─> Dispatches init_model() call to all workers via Ray RPC

E. verl/workers/fsdp_workers.py (lines 1270-1273) OR
   verl/workers/megatron_workers.py (lines 904-946)
   └─> RewardModelWorker.init_model() method
       └─> Calls: self._build_model(config) or self._build_rm_model()

F. Model Checkpoint Loading:

   For FSDP strategy (fsdp_workers.py, lines 1193-1267):
   └─> RewardModelWorker._build_model() (line 1193)
       ├─> Line 1200: copy_to_local(config.model.path)  # Download checkpoint
       ├─> Line 1211: AutoConfig.from_pretrained(local_path)
       └─> Line 1220-1226: AutoModelForTokenClassification.from_pretrained(
               pretrained_model_name_or_path=local_path,  # ← CHECKPOINT LOADED HERE
               config=model_config,
               torch_dtype=torch.bfloat16,
           )
       └─> Wraps model with FSDP/FSDP2 (lines 1242-1266)

   For Megatron strategy (megatron_workers.py, lines 861-901):
   └─> RewardModelWorker._build_rm_model() (line 861)
       ├─> Line 866: _init_hf_config_and_tf_config(model_path, ...)
       ├─> Lines 883-888: get_model() creates Megatron model
       └─> Lines 893-897: Load checkpoint:
           ├─> If use_dist_checkpointing:
           │   └─> load_mcore_dist_weights(reward_model, dist_checkpointing_path)
           └─> Else:
               └─> load_megatron_gptmodel_weights(config, hf_config, reward_model)
```

### 2. **Usage Flow** (Model Inference)

```
A. verl/trainer/ppo/ray_trainer.py (lines 1018-1022)
   └─> During training loop in fit() method:
       if self.use_rm:
           reward_tensor = self.rm_wg.compute_rm_score(batch)
           batch = batch.union(reward_tensor)

B. verl/single_controller/ray/base.py (WorkerGroup.compute_rm_score)
   └─> Dispatches compute_rm_score() call to all workers via Ray RPC

C. verl/workers/fsdp_workers.py (lines 1405-1462) OR
   verl/workers/megatron_workers.py (lines 951-958)
   └─> RewardModelWorker.compute_rm_score(data: DataProto) method

   For FSDP strategy (fsdp_workers.py):
   └─> Line 1405: compute_rm_score() entry point
       ├─> Lines 1412-1423: Prepare input data (switch chat template if needed)
       ├─> Lines 1429-1443: Process micro-batches:
       │   └─> For each micro_batch:
       │       └─> Line 1441: rm_score = self._forward_micro_batch(micro_batch)
       ├─> Line 1275: _forward_micro_batch() method
       │   ├─> Line 1306 or 1317: output = self.reward_module(...)  # ← MODEL FORWARD PASS
       │   │   └─> reward_module is the loaded AutoModelForTokenClassification
       │   └─> Line 1323: Extract score from last valid token
       └─> Line 1451: Expand to token-level scores
       └─> Returns DataProto with "rm_scores" tensor

   For Megatron strategy (megatron_workers.py):
   └─> Line 951: compute_rm_score() entry point
       ├─> Line 956: output = self.rm.compute_reward(data)
       └─> verl/workers/reward_model/megatron/reward_model.py
           └─> MegatronRewardModel.compute_reward() method
               └─> Calls reward_model_module forward pass
```

## Key Files Summary

| File | Purpose | Key Methods |
|------|---------|-------------|
| `verl/trainer/main_ppo.py` | Entry point, selects worker class | Lines 149-156: Import RewardModelWorker |
| `verl/trainer/ppo/ray_trainer.py` | Trainer orchestrates workers | Line 754: Create worker class<br>Line 784: Call init_model()<br>Line 1021: Call compute_rm_score() |
| `verl/workers/fsdp_workers.py` | FSDP reward model worker | Line 1158: RewardModelWorker class<br>Line 1270: init_model()<br>Line 1193: _build_model() - loads checkpoint<br>Line 1405: compute_rm_score()<br>Line 1275: _forward_micro_batch() |
| `verl/workers/megatron_workers.py` | Megatron reward model worker | Line 821: RewardModelWorker class<br>Line 904: init_model()<br>Line 861: _build_rm_model() - loads checkpoint<br>Line 951: compute_rm_score() |
| `verl/workers/reward_model/megatron/reward_model.py` | Megatron reward model wrapper | Line 34: MegatronRewardModel class<br>compute_reward() method |

## Checkpoint Loading Details

**FSDP Strategy:**
- Checkpoint path: `config.reward_model.model.path`
- Loading happens in: `fsdp_workers.py`, line 1220
- Method: `AutoModelForTokenClassification.from_pretrained()`
- Model type: HuggingFace AutoModelForTokenClassification with `num_labels=1`

**Megatron Strategy:**
- Checkpoint path: `config.reward_model.model.path` or `config.megatron.dist_checkpointing_path`
- Loading happens in: `megatron_workers.py`, lines 893-897
- Methods: `load_mcore_dist_weights()` or `load_megatron_gptmodel_weights()`
- Model type: Megatron GPT model with value head

## Model Forward Pass

**FSDP:**
- Input: `input_ids`, `attention_mask`, `position_ids`
- Forward: `self.reward_module(input_ids, attention_mask, position_ids)`
- Output: `logits` shape `(batch_size, seq_len, 1)` → squeezed to `(batch_size, seq_len)`
- Extract: Score from last valid token position

**Megatron:**
- Input: `DataProto` with batch data
- Forward: Through `MegatronRewardModel.compute_reward()` → `reward_model_module` forward
- Output: Token-level reward scores

