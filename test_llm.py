import asyncio
from app.llm.provider import get_llm_provider

async def main():
    print("Initializing LLM provider...")
    try:
        provider = get_llm_provider()
        print("Active provider:", provider)
        print("Sending prompt...")
        ans = await provider.generate("Say hello, this is a test.", [])
        print("Answer:")
        print(ans)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(main())
