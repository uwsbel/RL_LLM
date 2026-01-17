
set -e
set -u

WANDB_TOKEN=60b3bda8c6fe6553f6091ed9ec277e7da90c004c
IF_THINK=True
export RAY_record_ref_creation_sites=1
export HYDRA_FULL_ERROR=1
export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
export LEARN_RATE=4e-7
export RAY_TEMP_DIR=/code/jingquaw-sandbox/tmp/ray
RUN_NAME="rlvr_small_grpo${BASE_MODEL//\//_}_${IF_THINK}_lr${LEARN_RATE}"
export check_point_path=/code/jingquaw-sandbox/RL_LLM/RLVER/code/checkpoints/${RUN_NAME}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

ray stop --force || true
ray start --head --temp-dir="$RAY_TEMP_DIR"
pip install torchdata
mkdir -p "$check_point_path"
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{
    "env_vars": {
      "WANDB_TOKEN": "'"$WANDB_TOKEN"'",
      "WANDB_API_KEY": "'"$WANDB_TOKEN"'",
      "PYTHONUNBUFFERED": "1",
      "HYDRA_FULL_ERROR": "1",
      "RAY_record_ref_creation_sites": "1"
    },
    "working_dir": "./",
    "pip": ["latex2sympy2", "word2number", "timeout_decorator"],
    "excludes": ["/checkpoints/*","saves/*"]
    }' -- PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    +data.virtual_dataset_size=8 \
    +data.val_virtual_dataset_size=8 \
    data.prompt_key=prompt \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=10000 \
    data.max_response_length=20000 \
    data.return_raw_chat=True \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.thinking=$IF_THINK \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=2e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.02 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=48000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.use_loss_generation_mask=False \
    actor_rollout_ref.rollout.name=vllm_multi_turn_via_chat \
    +actor_rollout_ref.rollout.environment.name=url_environment \
    +actor_rollout_ref.rollout.environment.per_turn_length=4096 \
    +actor_rollout_ref.rollout.environment.max_turns=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=48000 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=48000 \
    trainer.project_name=rlvr \
    trainer.experiment_name=$RUN_NAME \
    trainer.default_local_dir=$check_point_path \
    trainer.logger=['console','wandb'] \
    +trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.save_rollout=True \
    trainer.test_freq=5 \
    trainer.total_epochs=2000 \
    trainer.total_training_steps=2000 \
    2>&1 | tee -a "$check_point_path/train.log"