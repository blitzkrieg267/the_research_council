#!/usr/bin/env python3
"""
Test script to verify Azure OpenAI configuration works
"""
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

def test_azure_openai():
    try:
        # Configuration from user's documentation
        client = AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint="https://nachalo-2324-resource.cognitiveservices.azure.com/",
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )

        # Test call
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": "Hello, can you confirm this Azure OpenAI connection is working?",
                }
            ],
            max_tokens=100,
            temperature=0.7,
            model="gpt-4o-mini"  # deployment name
        )

        print("✅ Azure OpenAI connection successful!")
        print(f"Response: {response.choices[0].message.content}")
        return True

    except Exception as e:
        print(f"❌ Azure OpenAI connection failed: {e}")
        return False

if __name__ == "__main__":
    test_azure_openai()
