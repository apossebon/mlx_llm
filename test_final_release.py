from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, FunctionMessage, ChatMessage
import asyncio


GPT_MODEL_ID = "mlx-community/gpt-oss-20b-MXFP4-Q8"
Qwen_MODEL_ID = "lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit"

questions_test= [
    "Hello, how are you?",
    "What is the capital of France?",
    "Can you tell me a joke?",
    "What is the weather like today?",
    "Who won the World Cup in 2018?",
    "What is the meaning of life?",
    "How do I make a cake?",
    "What is the largest mammal on Earth?",
    "Can you recommend a good book?",
    "What is the speed of light?"
]

async def test_generate_async():
    from src.mlxchat.MLXChat import MLXChat
    
    model = MLXChat(model_name=Qwen_MODEL_ID, temperature=0.7, max_tokens=1024, use_gpt_harmony_response_format=False)
    if model.init():
        print("Modelo inicializado com sucesso.")

        for question in questions_test:
            response = await model.ainvoke([HumanMessage(content=question)])
            print(f"Q: {question}\nA: {response.content}\n")
            # print(f"Thinking: {response.get('thinking', False)}\n")

    else:
        print("Falha ao inicializar o modelo.")

def test_generate_sync():
    from src.mlxchat.MLXChat import MLXChat
    
    model = MLXChat(model_name=GPT_MODEL_ID, temperature=0.7, max_tokens=1024, use_gpt_harmony_response_format=True)
    if model.init():
        print("Modelo inicializado com sucesso.")

        for question in questions_test:
            response = model.invoke([HumanMessage(content=question)])
            print(f"Q: {question}\nA: {response.content}\n")
            # print(f"Thinking: {response.get('thinking', False)}\n")

    else:
        print("Falha ao inicializar o modelo.")

if __name__ == "__main__":
    # test_generate_sync()
    asyncio.run(test_generate_async())
