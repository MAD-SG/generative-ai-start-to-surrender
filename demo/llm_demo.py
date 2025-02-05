from lmdeploy import pipeline, PytorchEngineConfig

if __name__ == "__main__":
    pipe = pipeline("deepseek-ai/DeepSeek-V3-FP8", backend_config=PytorchEngineConfig(tp=8))
    messages_list = [
        [{"role": "user", "content": "Who are you?"}],
        [{"role": "user", "content": "Translate the following content into Chinese directly: DeepSeek-V3 adopts innovative architectures to guarantee economical training and efficient inference."}],
        [{"role": "user", "content": "Write a piece of quicksort code in C++."}],
    ]
    output = pipe(messages_list)
    print(output)