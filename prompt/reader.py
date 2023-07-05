import re
from pathlib import Path

class PromptReader:
    
    @staticmethod
    def read_agent_prompt(current_file: str, prompt_file: str) -> str:
        file_path = str(Path(current_file).resolve().parent) + "/prompt/" + prompt_file
        try:
            f = open(file_path, "r")
            file_content = f.read()
        except FileNotFoundError as e:
            print(e.__str__())
            raise e
        return file_content
    

    @staticmethod
    def clean_prompt(prompt) -> str:
        prompt = re.sub('[ \t]+', ' ', prompt)
        return prompt.strip()