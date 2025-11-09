"""
Custom Azure OpenAI LLM provider for CrewAI
"""
import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from openai import AzureOpenAI
from crewai.llm import LLM

load_dotenv()

class CustomAzureLLM(LLM):
    """
    Custom LLM class for Azure OpenAI that uses the AzureOpenAI client
    Inherits from CrewAI's LLM but overrides the client
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ):
        # Initialize the Azure OpenAI client
        self._azure_client = AzureOpenAI(
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )

        # Store parameters
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop

        # Initialize parent with dummy values to avoid provider detection
        super().__init__(
            model="dummy_model",
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            **kwargs
        )

    @property
    def client(self):
        """Override the client property to return our Azure client"""
        return self._azure_client

    def call(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """
        Make a call to Azure OpenAI
        """
        # Prepare parameters
        call_params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        # Add optional parameters if set
        if self.max_tokens:
            call_params["max_tokens"] = self.max_tokens
        if self.top_p:
            call_params["top_p"] = self.top_p
        if self.frequency_penalty:
            call_params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty:
            call_params["presence_penalty"] = self.presence_penalty
        if self.stop:
            call_params["stop"] = self.stop

        # Override with any kwargs passed to call
        call_params.update(kwargs)

        # Make the API call
        response = self.client.chat.completions.create(**call_params)

        return response.choices[0].message.content
