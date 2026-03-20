# RewardWorker and RewardManager Interaction

## Overview

**RewardWorker** (RewardModelWorker) and **RewardManager** are two separate components that work together to provide rewards for RL training:

1. **RewardWorker**: Computes **raw token-level scores** from a neural reward model
2. **RewardManager**: **Postprocesses** rewards (either from RewardWorker or rule-based) and returns final reward tensors

## Interaction Flow

### Step 1: RewardWorker Computes Raw Scores

```python
# verl/trainer/ppo/ray_trainer.py (line 1021-1022)
if self.use_rm:
    reward_tensor = self.rm_wg.compute_rm_score(batch)  # ← RewardWorker computes
    batch = batch.union(reward_tensor)  # ← Adds "rm_scores" to batch
```

**RewardWorker.compute_rm_score()** returns:
- `DataProto` with key `"rm_scores"` containing **token-level scores**
- Shape: `(batch_size, response_length)` - one score per token position
- These are **raw model outputs** from the reward model's forward pass

### Step 2: RewardManager Processes the Scores

```python
# verl/trainer/ppo/ray_trainer.py (line 1029)
reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
```

**RewardManager.__call__()** (in `verl/workers/reward_manager/naive.py`):

```python
# Lines 47-51
if "rm_scores" in data.batch.keys():
    if return_dict:
        return {"reward_tensor": data.batch["rm_scores"]}
    else:
        return data.batch["rm_scores"]
```

**Key Points:**
- If `rm_scores` exists (from RewardWorker), RewardManager **uses them directly**
- If `rm_scores` doesn't exist, RewardManager computes **rule-based rewards** using `compute_score()` function
- RewardManager can also **combine** model-based and rule-based rewards

### Step 3: Final Reward Tensor

The RewardManager returns:
- `reward_tensor`: Token-level reward tensor `(batch_size, response_length)`
- `reward_extra_infos_dict`: Additional metadata (optional)

This is then used in the PPO training loop:
```python
# verl/trainer/ppo/ray_trainer.py (line 1091)
batch.batch["token_level_scores"] = reward_tensor
```

## Summary

**Yes, your understanding is correct:**

1. ✅ **RewardWorker** provides **token-level raw rewards** (`rm_scores`)
2. ✅ **RewardManager** postprocesses these rewards (or computes rule-based ones)
3. ✅ RewardManager returns the final **reward tensor** used for training

**The flow:**
```
RewardWorker.compute_rm_score() 
  → Returns DataProto with "rm_scores" (raw token-level scores)
  → Merged into batch
  → RewardManager.__call__() 
    → Checks if "rm_scores" exists
    → If yes: uses them directly
    → If no: computes rule-based rewards
  → Returns final reward_tensor
```

---

## Model Class Options

### Current Implementation

**FSDP Strategy** (`verl/workers/fsdp_workers.py`):
- **Model Class**: `AutoModelForTokenClassification` (line 1160, 1669)
- **Comment**: "Note that we only implement the reward model that is subclass of AutoModelForTokenClassification"
- **Output**: Token-level logits `(batch_size, seq_len, num_labels)` where `num_labels=1`
- **Usage**: Extracts score from last valid token position

**Megatron Strategy** (`verl/workers/megatron_workers.py`):
- **Model Class**: Custom Megatron GPT model with value head (NOT AutoModelForTokenClassification)
- **Comment**: "Note that we only implement the reward model that is subclass of AutoModelForSequenceClassification" (line 823)
- **Output**: Sequence-level scores via `MegatronRewardModel.compute_reward()`
- **Usage**: Uses Megatron's value head architecture

### Flexibility

**The code is somewhat restrictive but not completely:**

1. **FSDP Strategy**: Currently hardcoded to `AutoModelForTokenClassification`
   - Line 1641: `from transformers import AutoConfig, AutoModelForTokenClassification`
   - Line 1669: `reward_module = AutoModelForTokenClassification.from_pretrained(...)`
   - **To use a different class**: You would need to modify `_build_model()` method

2. **Megatron Strategy**: More flexible
   - Uses custom model provider function
   - Can work with different architectures via `init_mcore_model()`

3. **Why TokenClassification?**
   - Provides **token-level scores** which are needed for token-level RL training
   - Each token position gets a score, allowing fine-grained reward shaping
   - The score is extracted from the last valid token (EOS position)

### Alternative Approaches

If you want to use a different model class:

1. **Modify FSDP RewardModelWorker._build_model()**:
   ```python
   # Instead of AutoModelForTokenClassification
   from transformers import AutoModelForSequenceClassification
   reward_module = AutoModelForSequenceClassification.from_pretrained(...)
   # Then adapt the forward pass to extract token-level scores
   ```

2. **Use Megatron Strategy**: More flexible architecture support

3. **Custom RewardWorker**: Create your own worker class that inherits from `Worker` and implements `compute_rm_score()`

### Recommendation

- **For token-level rewards**: `AutoModelForTokenClassification` is the standard choice
- **For sequence-level rewards**: You could use `AutoModelForSequenceClassification` but would need to adapt the code to expand sequence-level scores to token-level
- **For maximum flexibility**: Use Megatron strategy or create a custom worker

**Bottom line**: The current implementation is optimized for `AutoModelForTokenClassification` in FSDP, but the architecture allows for modification if needed.

