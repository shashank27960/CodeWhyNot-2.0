import ollama

class OllamaCounterfactualPromptGenerator:
    def __init__(self, model_name="codellama:latest", host="http://localhost:11434"):
        self.model_name = model_name
        self.host = host

    def build_prompt(self, original_prompt, n=3):
        system_instruction = (
            "You are an expert in code prompt engineering. "
            "Given a code generation prompt, generate a list of alternative prompts that solve the same problem "
            "but use a different algorithmic approach or concept. Each alternative should be a minimal, meaningful change.\n"
        )
        user_instruction = f"Original prompt: '{original_prompt}'\nList {n} alternative prompts:"
        return system_instruction + user_instruction

    def parse_alternatives(self, text):
        lines = text.split("\n")
        alternatives = []
        for line in lines:
            if line.strip().startswith(("1.", "2.", "3.", "-", "*")):
                alt = line.split(".", 1)[-1] if "." in line else line
                alt = alt.lstrip("-* ").strip()
                if alt:
                    alternatives.append(alt)
            elif line.strip():
                alternatives.append(line.strip())
        return alternatives

    def generate_counterfactuals(self, original_prompt, n=3):
        prompt = self.build_prompt(original_prompt, n)
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        text = response['message']['content']
        alternatives = self.parse_alternatives(text)
        return alternatives[:n] 