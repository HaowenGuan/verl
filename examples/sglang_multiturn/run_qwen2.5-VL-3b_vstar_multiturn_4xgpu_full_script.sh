set -x
export HYDRA_FULL_ERROR=1
ulimit -n 65535

# ---- new: clean up Ray/zombies to avoid leftover ports ----
ray stop --force || true
pkill -9 -u "$USER" -f "python.*sglang" || true

# ---- new: pick a stable MASTER_ADDR/PORT for torch.distributed ----
export MASTER_ADDR="$(hostname -s)"
# choose a high random port; you can also hardcode one in a range you control
export MASTER_PORT=$(( 10000 + RANDOM % 50000 ))

# (optional but helpful) NCCL & network hygiene
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker0    # use real NICs, skip loopback/docker
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# (optional) avoid Ray dashboard binding ports
export RAY_DISABLE_DASHBOARD=1
export RAY_memory_monitor_refresh_ms=0

# Log where we are and what ports we use
hostname
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='vstar_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_files=/home/haowen.guan001/data/V-STaR/vstar_train_temporal_sample_image.parquet \
    data.val_files=/home/haowen.guan001/data/V-STaR/vstar_val_temporal_sample_image.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=4096 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='vstar_async_rl' \
    trainer.experiment_name='qwen2.5-3b_function_rm-vstar-sgl-multi-w-tool-verify-n16-4cards' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    trainer.total_epochs=150 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    critic.ppo_max_token_len_per_gpu=8192 \
    critic.forward_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    critic.model.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/video_sample_tool_config.yaml" \
    $@


#python3 -m verl.trainer.main_ppo \
#    --config-path="$CONFIG_PATH" \
#    --config-name='vstar_multiturn_grpo' \
#    algorithm.adv_estimator=grpo \
#    data.train_files=/home/haowen.guan001/data/V-STaR/vstar_train_temporal_image.parquet \
#    data.val_files=/home/haowen.guan001/data/V-STaR/vstar_val_temporal_image.parquet \
#    data.train_batch_size=64 \
#    data.max_prompt_length=25000 \
#    data.max_response_length=1024 \
#    data.filter_overlong_prompts=True \
#    data.truncation='error' \
#    data.return_raw_chat=True \
#    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
#    actor_rollout_ref.actor.optim.lr=1e-6 \
#    actor_rollout_ref.model.use_remove_padding=True \
#    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
#    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
#    actor_rollout_ref.actor.use_kl_loss=True \
#    actor_rollout_ref.actor.kl_loss_coef=0.001 \
#    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#    actor_rollout_ref.actor.entropy_coeff=0 \
#    actor_rollout_ref.model.enable_gradient_checkpointing=True \
#    actor_rollout_ref.actor.fsdp_config.param_offload=False \
#    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
#    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
#    actor_rollout_ref.rollout.name=sglang \
#    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
#    actor_rollout_ref.rollout.n=16 \
#    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
#    actor_rollout_ref.ref.fsdp_config.param_offload=True \
#    algorithm.use_kl_in_reward=False \
#    trainer.critic_warmup=0 \
#    trainer.logger='["console"]' \
#    trainer.project_name='vstar_async_rl' \
#    trainer.experiment_name='qwen2.5-3b_function_rm-vstar-sgl-multi-w-tool-verify-n16-4cards' \
#    trainer.n_gpus_per_node=2 \
#    trainer.nnodes=1 \
#    trainer.save_freq=-1 \
#    trainer.test_freq=20 \
#    trainer.total_epochs=15 \
#    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
#    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
#    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
#    critic.ppo_max_token_len_per_gpu=8192 \
#    critic.forward_max_token_len_per_gpu=8192 \
#    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
#    critic.model.fsdp_config.model_dtype=bfloat16 \
#    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/vstar_tool_config.yaml" \
#    $@


#python3 -m verl.trainer.main_ppo \
#    algorithm.adv_estimator=grpo \
#    data.train_files=/home/haowen.guan001/data/V-STaR/vstar_train_temporal.parquet \
#    data.val_files=/home/haowen.guan001/data/V-STaR/vstar_val_temporal.parquet \
#    data.train_batch_size=8 \
#    data.max_prompt_length=25000 \
#    data.max_response_length=200 \
#    data.filter_overlong_prompts=True \
#    data.truncation='error' \
#    data.video_key=videos \
#    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
#    actor_rollout_ref.actor.optim.lr=1e-6 \
#    actor_rollout_ref.model.use_remove_padding=True \
#    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
#    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
#    actor_rollout_ref.actor.use_kl_loss=True \
#    actor_rollout_ref.actor.kl_loss_coef=0.01 \
#    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#    actor_rollout_ref.actor.entropy_coeff=0 \
#    actor_rollout_ref.model.enable_gradient_checkpointing=False \
#    actor_rollout_ref.actor.use_torch_compile=False \
#    actor_rollout_ref.ref.use_torch_compile=False \
#    actor_rollout_ref.actor.fsdp_config.param_offload=False \
#    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
#    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
#    actor_rollout_ref.rollout.name=$ENGINE \
#    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
#    actor_rollout_ref.rollout.enable_chunked_prefill=False \
#    actor_rollout_ref.rollout.enforce_eager=False \
#    actor_rollout_ref.rollout.free_cache_engine=False \
#    actor_rollout_ref.rollout.n=1 \
#    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
#    actor_rollout_ref.ref.fsdp_config.param_offload=False \
#    algorithm.use_kl_in_reward=False \
#    trainer.critic_warmup=0 \
#    trainer.logger=['console','wandb'] \
#    trainer.project_name='verl_grpo_vstar' \
#    trainer.experiment_name='qwen2_5_vl_3b_vstar' \
#    trainer.n_gpus_per_node=8 \
#    trainer.nnodes=1 \
#    trainer.save_freq=200 \
#    trainer.val_before_train=False \
#    trainer.test_freq=20 \
#    trainer.total_epochs=10 $@