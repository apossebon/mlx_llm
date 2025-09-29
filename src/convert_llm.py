from mlx_lm import convert
from pathlib import Path

if __name__ == "__main__":

    target_path = "possebon--Qwen3-Omni-30B-A3B-Instruct-MLX-4bit"
        # Only convert if the target path does not already exist
    if not Path(target_path).exists():
        # For uniform 4-bit quantization, just use q_bits/q_group_size
        convert(
            hf_path="Qwen/Qwen3-Omni-30B-A3B-Instruct",
            mlx_path=target_path,
            quantize=True,
            q_group_size=64,
            q_bits=4,
            upload_repo=target_path
        )
            # If you want mixed per-layer quantization instead, pass a callable:
            # convert(hf_path="openai/gpt-oss-20b", mlx_path=target_path, quantize=True, quant_predicate=mixed_quantization)