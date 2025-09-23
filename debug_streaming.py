#!/usr/bin/env python3
"""
Debug script to test streaming functionality
"""
import asyncio
from langchain_core.messages import HumanMessage
from src.chatmlx_gpt import ChatMLX

async def test_streaming():
    """Test both Harmony and regular streaming"""
    
    print("Testing streaming functionality...")
    print("=" * 50)
    
    # Test 1: Regular format
    print("Test 1: Regular format streaming")
    try:
        llm_regular = ChatMLX(max_tokens=100, use_gpt_harmony_response_format=False)
        llm_regular.init()
        
        messages = [HumanMessage(content="Hello, how are you?")]
        
        print("Starting regular stream...")
        chunks = []
        async for chunk in llm_regular.astream(messages):
            print(f"Chunk: {chunk}")
            chunks.append(chunk)
        
        print(f"Total chunks received: {len(chunks)}")
        if chunks:
            print(f"First chunk: {chunks[0]}")
        else:
            print("❌ No chunks received in regular format!")
            
    except Exception as e:
        print(f"❌ Error in regular format: {e}")
    
    print("\n" + "=" * 50)
    
    # Test 2: Harmony format
    print("Test 2: Harmony format streaming")
    try:
        llm_harmony = ChatMLX(max_tokens=100, use_gpt_harmony_response_format=True)
        llm_harmony.init()
        
        messages = [HumanMessage(content="Hello, how are you?")]
        
        print("Starting Harmony stream...")
        chunks = []
        async for chunk in llm_harmony.astream(messages):
            print(f"Chunk: {chunk}")
            chunks.append(chunk)
        
        print(f"Total chunks received: {len(chunks)}")
        if chunks:
            print(f"First chunk: {chunks[0]}")
        else:
            print("❌ No chunks received in Harmony format!")
            
    except Exception as e:
        print(f"❌ Error in Harmony format: {e}")

if __name__ == "__main__":
    asyncio.run(test_streaming())
