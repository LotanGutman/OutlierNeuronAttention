import torch
from benchmarks.benchmark_induction import run_induction_experiment, InductionExperimentConfig
from src.config import ModelConfig

config = InductionExperimentConfig(
    model_config=ModelConfig(
        d_model=128,  # Increased so d_head=32
        num_heads=4,
        num_layers=4,
        r=16,         # Exact capacity we need (16 > 13)
        use_rope=True
    )
)

print("Starting test with r=16...")
# We can just test HOFA
from benchmarks.benchmark_MQAR import GenericBenchmarkLM, AttentionType
from benchmarks.benchmark_induction import train_induction

device = 'cuda'
model = GenericBenchmarkLM(
    vocab_size=config.vocab_size,
    d_model=config.model_config.d_model,
    attn_type=AttentionType.HOFA,
    num_heads=config.model_config.num_heads,
    num_layers=config.model_config.num_layers,
    model_cfg=config.model_config
).to(device)

history = train_induction(model, config, "HOFA (r=16)")
print("Final accuracy:", history['acc'][-1])
