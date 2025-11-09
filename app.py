import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from pathlib import Path
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

def extract_text_from_file(uploaded_file):
    """Extract text content from uploaded file"""
    file_extension = Path(uploaded_file.name).suffix.lower()

    # Create a temporary file to save the uploaded content
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    try:
        if file_extension == '.pdf':
            # Use pypdf to extract text from PDF
            from pypdf import PdfReader
            reader = PdfReader(temp_file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text

        elif file_extension in ['.docx', '.doc']:
            # Use python-docx to extract text from Word documents
            from docx import Document
            doc = Document(temp_file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text

        elif file_extension == '.txt':
            # Read plain text files
            with open(temp_file_path, 'r', encoding='utf-8') as f:
                return f.read()

        else:
            st.error(f"Unsupported file type: {file_extension}. Please upload PDF, DOCX, or TXT files.")
            return ""

    except Exception as e:
        st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
        return ""
    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)

def summarize_with_azure(content, topic=""):
    """Summarize content using Azure OpenAI"""
    try:
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )

        # Prepare the prompt
        prompt = f"Please provide a comprehensive summary of the following content"
        if topic:
            prompt += f" related to the topic: {topic}"
        prompt += f".\n\nContent:\n{content[:8000]}"  # Limit content length

        # Make the API call
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides clear, concise summaries of research content."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating summary: {str(e)}"

def main():
    st.title("📄 Document Summarizer")
    st.markdown("Upload documents and get AI-powered summaries using Azure OpenAI")

    # Check if Azure credentials are configured
    if not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_ENDPOINT"):
        st.error("Azure OpenAI credentials not configured. Please check your .env file.")
        return

    st.header("📤 Upload Documents")

    # File upload section
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT files",
        type=["pdf", "docx", "doc", "txt"],
        accept_multiple_files=True,
        help="Upload research papers, articles, or documents to summarize"
    )

    # Topic input
    topic = st.text_input(
        "Topic (Optional)",
        placeholder="e.g., Artificial Intelligence, Healthcare, Machine Learning",
        help="Specify a topic to focus the summary on"
    )

    # Display uploaded files
    if uploaded_files:
        st.success(f"📄 {len(uploaded_files)} file(s) uploaded successfully!")

        for i, file in enumerate(uploaded_files):
            with st.expander(f"📄 {file.name} ({file.size} bytes)"):
                # Show preview of content
                file_content = extract_text_from_file(file)
                if file_content:
                    preview_length = min(500, len(file_content))
                    st.text_area(
                        f"Content Preview ({len(file_content)} characters total)",
                        file_content[:preview_length] + ("..." if len(file_content) > preview_length else ""),
                        height=150,
                        disabled=True,
                        key=f"preview_{i}"
                    )

    # Summarize button
    if st.button("🚀 Generate Summary", type="primary", disabled=not uploaded_files):
        if not uploaded_files:
            st.error("Please upload at least one document first.")
            return

        with st.spinner("🔄 Processing documents and generating summary..."):
            # Extract text from all uploaded files
            all_content = ""
            for file in uploaded_files:
                st.write(f"📖 Processing {file.name}...")
                file_content = extract_text_from_file(file)
                if file_content:
                    all_content += f"\n--- {file.name} ---\n{file_content}\n"

            if not all_content.strip():
                st.error("No readable content found in uploaded files.")
                return

            # Generate summary
            st.write("🤖 Generating AI summary...")
            summary = summarize_with_azure(all_content, topic)

            # Display results
            st.success("✅ Summary generated successfully!")

            st.header("📋 Summary Results")
            st.markdown(f"**Topic:** {topic if topic else 'General'}")
            st.markdown(f"**Documents processed:** {len(uploaded_files)}")
            st.markdown(f"**Total content length:** {len(all_content)} characters")

            st.subheader("📝 AI-Generated Summary")
            st.markdown(summary)

            # Option to download summary
            st.download_button(
                label="📥 Download Summary",
                data=summary,
                file_name="document_summary.txt",
                mime="text/plain",
                help="Download the summary as a text file"
            )

    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This tool uses Azure OpenAI to provide intelligent summaries of your documents.

        **Supported formats:**
        - 📄 PDF documents
        - 📝 Word documents (.docx, .doc)
        - 📃 Plain text files (.txt)

        **Features:**
        - Multi-document processing
        - Topic-focused summaries
        - Content preview
        - Downloadable results
        """)

        st.header("⚙️ Configuration")
        st.markdown("Using Azure OpenAI GPT-4o-mini")
        if os.getenv("AZURE_OPENAI_ENDPOINT"):
            endpoint_display = os.getenv("AZURE_OPENAI_ENDPOINT").replace("https://", "").split(".")[0]
            st.text(f"Endpoint: {endpoint_display}")

if __name__ == "__main__":
    main()
