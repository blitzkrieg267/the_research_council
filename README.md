# 📄 Document Summarizer

A simple web application that allows you to upload documents and get AI-powered summaries using Azure OpenAI.

## Features

- **Document Upload**: Support for PDF, DOCX, and TXT files
- **Text Extraction**: Automatic text extraction from uploaded documents
- **AI Summarization**: Uses Azure OpenAI GPT-4o-mini for intelligent summaries
- **Topic Focus**: Optional topic specification for focused summaries
- **Content Preview**: Preview extracted content before summarization
- **Download Results**: Download summaries as text files

## Installation

1. Install required dependencies:
```bash
pip install streamlit openai python-docx pypdf
```

2. Configure Azure OpenAI credentials in `.env`:
```bash
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Upload your documents (PDF, DOCX, or TXT files)

3. Optionally specify a topic to focus the summary

4. Click "Generate Summary" to process the documents and create an AI summary

5. Download the summary if needed

## Supported File Formats

- **PDF**: Portable Document Format files
- **DOCX/DOC**: Microsoft Word documents
- **TXT**: Plain text files

## Azure OpenAI Configuration

The application uses Azure OpenAI with the following configuration:
- **Model**: GPT-4o-mini
- **API Version**: 2024-12-01-preview
- **Endpoint**: Your Azure OpenAI resource endpoint

Make sure your Azure OpenAI resource has the GPT-4o-mini model deployed and accessible.
