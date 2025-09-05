import ollama

def suggest_interventions_ml(concept, prompt, model_name="codellama:latest"):
    """
    Use an LLM to suggest alternative strategies for a given concept in the context of a prompt.
    Returns a list of alternative concepts/interventions.
    """
    system_prompt = (
        f"Given the concept '{concept}' in the following prompt, suggest alternative programming strategies or concepts that could be used instead. "
        "Return a comma-separated list."
    )
    user_prompt = f"Prompt: '{prompt}'\nAlternatives:"
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": system_prompt + user_prompt}],
        stream=False
    )
    text = response['message']['content']
    return [c.strip() for c in text.split(',') if c.strip()] 