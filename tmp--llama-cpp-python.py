import argparse

from llama_cpp import Llama

from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any


class LlamaLLM(LLM):
    model_path: str
    llm: Llama

    @property
    def _llm_type(self) -> str:
        return "llama-cpp-python"

    def __init__(self, model_path: str, **kwargs: Any):
        model_path = model_path
        llm = Llama(model_path=model_path)
        super().__init__(model_path=model_path, llm=llm, **kwargs)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.llm(prompt, stop=stop or [])
        return response["choices"][0]["text"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_path": self.model_path}


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, 
                    default= '/home/khan/Insync/alpha.khan@gmail.com/Google Drive/AI LLAMA Models/gpt4all-lora-q-converted.bin')
                    # '/home/khan/Documents/Github/AI-MODELS/llama/models/7B/ggml-model-q4_0.bin')
                    # "../models/7B/ggml-models.bin")
args = parser.parse_args()

# Load the model
llm = LlamaLLM(model_path=args.model)

# Basic Q&A
answer = llm(
    "Question: What is the capital of France? Answer: ", stop=["Question:", "\n"]
)
print(f"Answer: {answer.strip()}")

input("Press enter to continue (1/2)...")

# Using in a chain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

prompt = PromptTemplate(
    input_variables=["product"],
    template="\n\n### Instruction:\nWrite a good name for a company that makes {product}\n\n### Response:\n",
)
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain only specifying the input variable.
print(chain.run("colorful socks"))
