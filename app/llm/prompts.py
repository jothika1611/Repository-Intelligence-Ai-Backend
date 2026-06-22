class PromptBuilder:
    @staticmethod
    def build_prompt(question: str, context: list[str]) -> str:
        """
        Safely builds a unified system prompt, injecting explicit boundaries 
        around retrieved context to prevent prompt injection leakage.
        """
        context_str = "\n\n".join(context)
        
        return f"""You are an expert AI assistant designed to answer questions about a GitHub repository.
You must strictly base your answer ONLY on the provided context.
If the context does not contain the answer, say "I do not have enough information to answer this based on the repository content."
Do NOT invent citations, external links, or hallucinate code.

### CONTEXT START
{context_str}
### CONTEXT END

Question: {question}

Answer in markdown format:"""
