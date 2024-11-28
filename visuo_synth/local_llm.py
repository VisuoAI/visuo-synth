from typing import Any, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.language_models.llms import LLM
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration


class QwenCoder(LLM):
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    temperature: float = 0.7
    max_length: int = 2048
    tokenizer: Optional[Any] = None
    model: Optional[Any] = None
    device: str = "cpu"  # Default to CPU, will be updated in __init__

    def __init__(self):
        super().__init__()

        # Determine the best available device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"  # Apple Silicon GPU
        else:
            self.device = "cpu"

        print(f"Loading model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        # Load model with float16 for better memory efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        print("Model loaded successfully!")

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Move inputs to appropriate device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_length=self.max_length,
            temperature=self.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from the response
        if response.startswith(prompt):
            response = response[len(prompt) :].strip()
        return response

    def invoke(self, input: List[BaseMessage], **kwargs: Any) -> ChatResult:
        if not input:
            raise ValueError("input messages is empty")

        # Combine all messages into a single prompt
        prompt = "\n".join(
            msg.content for msg in input if isinstance(msg, HumanMessage)
        )

        # Generate response
        response = self._call(prompt)

        # Create ChatGeneration object with AIMessage
        gen = ChatGeneration(message=AIMessage(content=response))

        # Return ChatResult
        return ChatResult(generations=[gen])

    @property
    def _llm_type(self) -> str:
        return "qwen-coder"
