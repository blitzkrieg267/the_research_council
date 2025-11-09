import os
from pathlib import Path
from typing import Dict, Any, List
from openai import AzureOpenAI
import requests
from datetime import datetime

class ResearchAgent:
    def __init__(self):
        self.client = AzureOpenAI(
            api_version="2025-03-01-preview",
            azure_endpoint="https://nachalo-2324-resource.cognitiveservices.azure.com/",
            api_key="os.getenv("AZURE_OPENAI_API_KEY", "your-api-key-here")",
        )

    def search_arxiv(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search arXiv for relevant papers."""
        try:
            # arXiv API search
            base_url = "http://export.arxiv.org/api/query"
            params = {
                'search_query': query,
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }

            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                # Parse XML response (simplified)
                papers = []
                # This is a simplified parser - in production you'd use xml.etree
                content = response.text
                entries = content.split('<entry>')[1:]  # Skip first part

                for entry in entries[:max_results]:
                    try:
                        title = entry.split('<title>')[1].split('</title>')[0] if '<title>' in entry else "Unknown"
                        authors = []
                        if '<author>' in entry:
                            author_parts = entry.split('<author>')
                            for part in author_parts[1:]:
                                if '<name>' in part:
                                    name = part.split('<name>')[1].split('</name>')[0]
                                    authors.append(name)

                        abstract = entry.split('<summary>')[1].split('</summary>')[0] if '<summary>' in entry else ""
                        link = entry.split('<id>')[1].split('</id>')[0] if '<id>' in entry else ""

                        papers.append({
                            'title': title,
                            'authors': authors,
                            'abstract': abstract,
                            'link': link,
                            'source': 'arXiv'
                        })
                    except:
                        continue

                return papers
            else:
                return []
        except Exception as e:
            print(f"arXiv search error: {e}")
            return []

    def search_wikipedia(self, query: str) -> Dict[str, Any]:
        """Search Wikipedia for relevant information."""
        try:
            # Wikipedia API
            base_url = "https://en.wikipedia.org/api/rest_v1/page/summary"
            response = requests.get(f"{base_url}/{query.replace(' ', '_')}")

            if response.status_code == 200:
                data = response.json()
                return {
                    'title': data.get('title', ''),
                    'extract': data.get('extract', ''),
                    'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                    'source': 'Wikipedia'
                }
            else:
                return {}
        except Exception as e:
            print(f"Wikipedia search error: {e}")
            return {}

    def analyze_findings_question_based(self, documents: List[Dict], research_topic: str, iteration: int = 1, previous_context: str = "") -> Dict[str, Any]:
        """
        Analyze research findings using 10 comprehensive questions with detailed effort logging.

        Args:
            documents: List of research documents
            research_topic: The research topic
            iteration: Current iteration number
            previous_context: Context from previous iterations

        Returns:
            Question-based analysis results
        """
        # Initialize effort tracking
        effort_details = {
            'stage': 'content_processing',
            'task': 'compiling_research_content',
            'documents_received': len(documents),
            'iteration': iteration,
            'topic': research_topic,
            'start_time': datetime.now().isoformat(),
            'content_compilation': {},
            'question_analysis': {}
        }

        # Compile all content with length limits
        all_content = ""
        sources = []
        max_total_content = 5000  # Limit total content to stay within token limits

        for doc in documents:
            if 'abstract' in doc:
                doc_content = f"\nTitle: {doc.get('title', 'Unknown')}\nAbstract: {doc['abstract']}\n"
                # Truncate individual abstracts if too long
                if len(doc_content) > 1000:
                    doc_content = doc_content[:1000] + "...\n"
                all_content += doc_content
                sources.append({
                    'title': doc.get('title', 'Unknown'),
                    'authors': doc.get('authors', []),
                    'source': doc.get('source', 'Unknown'),
                    'link': doc.get('link', '')
                })
            elif 'extract' in doc:
                doc_content = f"\nTitle: {doc.get('title', 'Unknown')}\nContent: {doc['extract']}\n"
                # Truncate individual extracts if too long
                if len(doc_content) > 1500:
                    doc_content = doc_content[:1500] + "...\n"
                all_content += doc_content
                sources.append({
                    'title': doc.get('title', 'Unknown'),
                    'source': doc.get('source', 'Unknown'),
                    'link': doc.get('url', '')
                })

            # Stop adding content if we've reached the limit
            if len(all_content) >= max_total_content:
                break

        # Final truncation of total content
        if len(all_content) > max_total_content:
            all_content = all_content[:max_total_content] + "\n[Content truncated for token limits]"

        if not all_content.strip():
            return {
                'question_answers': {},
                'sources': sources,
                'status': 'no_content'
            }

        # 10 comprehensive questions for research analysis
        questions = [
            "What is the information content and quality of the sources?",
            "What are the key takeaways and main findings?",
            "What methods and approaches were used in the research?",
            "What are the main conclusions and implications?",
            "What evidence supports the findings?",
            "What are the limitations and potential biases?",
            "How does this research relate to existing knowledge?",
            "What are the practical applications and real-world implications?",
            "What gaps or unanswered questions remain?",
            "What future research directions are suggested?"
        ]

        question_answers = {}

        # Update effort tracking for content compilation
        effort_details['content_compilation'] = {
            'total_content_length': len(all_content),
            'sources_compiled': len(sources),
            'content_truncated': len(all_content) < sum(len(doc.get('abstract', doc.get('extract', ''))) for doc in documents),
            'max_content_limit': max_total_content
        }

        # Answer each question with effort tracking
        effort_details['stage'] = 'question_analysis'
        effort_details['question_analysis'] = {
            'total_questions': len(questions),
            'questions_processed': 0,
            'successful_answers': 0,
            'failed_answers': 0,
            'question_details': []
        }

        for i, question in enumerate(questions, 1):
            question_start = datetime.now()

            prompt = f"""
You are a Research Analyst Agent in iteration {iteration}. Answer the following question about the research on "{research_topic}".

QUESTION {i}: {question}

RESEARCH MATERIALS:
{all_content[:3000]}  # Truncated for token limits

PREVIOUS CONTEXT FROM OTHER AGENTS:
{previous_context}

YOUR TASK:
Provide a detailed, evidence-based answer to this specific question. Be comprehensive but focused. Cite specific sources and provide reasoning. Consider the previous context from other agents in your analysis.

Answer format: Provide a clear, structured response that directly addresses the question.
"""

            question_effort = {
                'question_number': i,
                'question_text': question,
                'prompt_length': len(prompt),
                'start_time': question_start.isoformat(),
                'status': 'processing'
            }

            try:
                response = self.client.responses.create(
                    model="gpt-4o-mini",
                    input=prompt,
                    instructions=f"You are answering question {i} about research analysis. Be thorough, evidence-based, and consider previous agent context."
                )

                question_answers[f"question_{i}"] = {
                    'question': question,
                    'answer': response.output_text.strip(),
                    'question_number': i
                }

                question_effort.update({
                    'status': 'completed',
                    'answer_length': len(response.output_text),
                    'end_time': datetime.now().isoformat(),
                    'duration_seconds': (datetime.now() - question_start).total_seconds()
                })

                effort_details['question_analysis']['successful_answers'] += 1

            except Exception as e:
                question_answers[f"question_{i}"] = {
                    'question': question,
                    'answer': f"Error generating answer: {str(e)}",
                    'question_number': i
                }

                question_effort.update({
                    'status': 'failed',
                    'error': str(e),
                    'end_time': datetime.now().isoformat(),
                    'duration_seconds': (datetime.now() - question_start).total_seconds()
                })

                effort_details['question_analysis']['failed_answers'] += 1

            effort_details['question_analysis']['questions_processed'] += 1
            effort_details['question_analysis']['question_details'].append(question_effort)

        # Final effort summary
        effort_details.update({
            'stage': 'completed',
            'end_time': datetime.now().isoformat(),
            'total_duration_seconds': (datetime.now() - datetime.fromisoformat(effort_details['start_time'])).total_seconds()
        })

        return {
            'question_answers': question_answers,
            'sources': sources,
            'topic': research_topic,
            'iteration': iteration,
            'total_questions': len(questions),
            'content_length': len(all_content),
            'status': 'analyzed',
            'effort_log': effort_details
        }

    def analyze_findings(self, documents: List[Dict], research_topic: str) -> Dict[str, Any]:
        """Legacy method for backward compatibility."""
        return self.analyze_findings_question_based(documents, research_topic, iteration=1)

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m the_council___research_tool.research_agent <research_topic>")
        sys.exit(1)

    topic = sys.argv[1]
    agent = ResearchAgent()

    try:
        print(f"🔍 Researching topic: {topic}")

        # Search arXiv
        print("📚 Searching arXiv...")
        arxiv_papers = agent.search_arxiv(topic, max_results=3)

        # Search Wikipedia
        print("🌐 Searching Wikipedia...")
        wiki_info = agent.search_wikipedia(topic)

        # Combine sources
        all_sources = arxiv_papers
        if wiki_info:
            all_sources.append(wiki_info)

        print(f"📋 Found {len(all_sources)} sources")

        # Analyze findings
        print("🧠 Analyzing findings...")
        analysis = agent.analyze_findings(all_sources, topic)

        print("\n" + "="*60)
        print("RESEARCH ANALYSIS:")
        print("="*60)
        print(analysis['analysis'])

        print(f"\n📚 Sources analyzed: {len(analysis['sources'])}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
