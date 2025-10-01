from mlx_lm import server
import argparse


def main():
    host = "0.0.0.0"
    port = 8000

    args_namespace = argparse.Namespace(
        # model= "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit",
        model = "lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit",
        adapter_path=None,
        host=host,
        port=port,
        draft_model=None,
        num_draft_tokens=3,
        trust_remote_code=False,
        log_level="INFO",
        chat_template="",
        use_default_chat_template=False,
        temp=0.7,
        top_p=0.85,
        top_k=40,
        min_p=0.0,
        max_tokens=4098,
        chat_template_args={},
    )

    server.run(host, port, server.ModelProvider(args_namespace))

    print(f"Server running on {host}:{port}")


if __name__ == "__main__":
    main()