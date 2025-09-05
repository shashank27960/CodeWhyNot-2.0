import ollama

def extract_concepts_ml(prompt, model_name="codellama:latest"):
    """
    Use an LLM to extract specific, context-driven programming concepts or algorithmic strategies from a prompt.
    Returns a list of concepts. No hardcoded lists or fallbacks.
    Logs and returns the raw LLM output for debugging.
    """
    system_prompt = (
        "Extract all specific programming concepts, operations, and requirements mentioned in the following prompt. "
        "Return a comma-separated list of concepts. Be as specific as possible.\n"
        "\n"
        "Prompt: '{user_prompt}'\n"
        "Concepts:"
    )
    user_prompt = prompt
    full_prompt = system_prompt.replace('{user_prompt}', user_prompt)
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": full_prompt}],
        stream=False
    )
    text = response['message']['content']
    print(f"[LLM Concept Extraction Raw Output]: {text}")  # For debugging/logging
    return text  # Return raw output for further parsing in the app 