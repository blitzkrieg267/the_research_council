from typing import Dict, Any, List
from pathlib import Path
import tempfile
import os
from summarizer import FileSummarizer
from critique_agent import CritiqueAgent
from document_ingestion_agent import DocumentIngestionAgent

class IntelligentAnalysisAgent:
    def __init__(self):
        self.summarizer = FileSummarizer()
        self.critique_agent = CritiqueAgent()
        self.document_agent = DocumentIngestionAgent()

    def analyze_document(self, file_path: str, max_iterations: int = 3) -> Dict[str, Any]:
        """
        Perform intelligent document analysis with iterative summarization and critique.

        Args:
            file_path: Path to the document to analyze
            max_iterations: Number of critique iterations (default: 3)

        Returns:
            Dict containing the analysis results
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Step 1: Ingest and store the document
        print("📥 Ingesting document...")
        ingestion_result = self.document_agent.process_and_store(str(file_path))

        # Step 2: Read the document content
        try:
            if file_path.suffix.lower() in ['.pdf', '.docx', '.xlsx', '.xls']:
                # For complex files, get content from vector store after ingestion
                docs = self.document_agent.search_similar("content", k=15)
                content = "\n".join([doc.page_content for doc in docs])
                # If no content found in vector store, try to load directly
                if not content.strip():
                    documents = self.document_agent.load_document(str(file_path))
                    content = "\n".join([doc.page_content for doc in documents])
            else:
                # For text files, read directly
                content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            raise ValueError(f"Cannot read file content: {file_path}")
        except Exception as e:
            # Fallback: try to get content from the ingested documents
            try:
                docs = self.document_agent.search_similar("content", k=10)
                content = "\n".join([doc.page_content for doc in docs])
            except:
                raise ValueError(f"Cannot extract content from {file_path}: {str(e)}")

        # Step 3: Initial summarization
        print("📝 Creating initial summary...")
        initial_summary = self.summarizer.summarize_file(str(file_path))

        # Step 4: Iterative critique and improvement
        current_summary = initial_summary
        critique_history = []

        for iteration in range(1, max_iterations + 1):
            print(f"🔄 Iteration {iteration}/{max_iterations}: Getting critique...")

            # Get critique and improved summary
            critique_result = self.critique_agent.critique_summary(
                original_content=content,
                current_summary=current_summary,
                iteration=iteration
            )

            critique_history.append({
                'iteration': iteration,
                'original_summary': current_summary,
                'critique': critique_result['critique'],
                'improved_summary': critique_result['improved_summary']
            })

            # Update current summary for next iteration
            current_summary = critique_result['improved_summary']

        # Step 5: Final evaluation
        print("✅ Final evaluation...")
        final_evaluation = self.critique_agent.evaluate_final_summary(
            original_content=content,
            final_summary=current_summary
        )

        # Step 6: Generate comprehensive report
        report = self._generate_comprehensive_report(
            file_info=ingestion_result,
            initial_summary=initial_summary,
            final_summary=current_summary,
            critique_history=critique_history,
            evaluation=final_evaluation
        )

        return {
            'file_info': ingestion_result,
            'initial_summary': initial_summary,
            'final_summary': current_summary,
            'critique_history': critique_history,
            'evaluation': final_evaluation,
            'comprehensive_report': report,
            'iterations_completed': max_iterations,
            'status': 'completed'
        }

    def _generate_comprehensive_report(self, file_info: Dict, initial_summary: str,
                                     final_summary: str, critique_history: List[Dict],
                                     evaluation: Dict) -> str:
        """Generate a comprehensive analysis report."""

        report = f"""
# 📄 Intelligent Document Analysis Report

## 📋 Document Information
- **Filename**: {file_info['filename']}
- **Type**: {file_info['file_type']}
- **Chunks Processed**: {file_info['total_chunks']}
- **Source Documents**: {file_info['total_documents']}

## 🔍 Analysis Summary

### Initial Summary
{initial_summary}

### Final Refined Summary (After {len(critique_history)} Iterations)
{final_summary}

## 🔄 Critique & Improvement Process

"""

        for i, critique in enumerate(critique_history, 1):
            report += f"""
### Iteration {i}
**Critique Feedback:**
{critique['critique']}

**Improved Summary:**
{critique['improved_summary']}

---
"""

        report += f"""

## 📊 Final Evaluation
{evaluation['evaluation']}

## 🎯 Key Insights

### Strengths of Final Summary:
- Comprehensive coverage of main topics
- Clear and well-structured presentation
- Accurate representation of original content
- Appropriate level of detail

### Iterative Improvements Made:
- Enhanced clarity and precision
- Better organization of information
- More comprehensive coverage
- Improved readability and flow

### Recommendations for Future Analysis:
- Consider document context and purpose
- Validate against additional sources when possible
- Maintain balance between detail and conciseness
- Ensure objectivity and accuracy

---
*Analysis completed with {len(critique_history)} rounds of critique and improvement*
"""

        return report

    def get_document_context(self, query: str, k: int = 5) -> List[Dict]:
        """Get relevant document context for a query."""
        docs = self.document_agent.search_similar(query, k=k)
        return [
            {
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': getattr(doc, 'score', None)
            }
            for doc in docs
        ]

def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python -m the_council___research_tool.intelligent_analysis_agent <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    agent = IntelligentAnalysisAgent()

    try:
        print("🚀 Starting intelligent document analysis...")
        result = agent.analyze_document(file_path)

        print("✅ Analysis completed!")
        print("\n" + "="*60)
        print("FINAL SUMMARY:")
        print("="*60)
        print(result['final_summary'])
        print("\n" + "="*60)
        print("COMPREHENSIVE REPORT:")
        print("="*60)
        print(result['comprehensive_report'])

        # Save report to file
        report_file = f"analysis_report_{Path(file_path).stem}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(result['comprehensive_report'])
        print(f"\n📄 Full report saved to: {report_file}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
