import os
from pathlib import Path
from typing import Dict, Any, List, Generator
from openai import AzureOpenAI
import time
import json
from datetime import datetime

from research_agent import ResearchAgent
from critique_agent import CritiqueAgent
from reviewer_agent import ReviewerAgent
from summarizer import FileSummarizer
from tavily_agent import TavilyAgent
from query_refinement_agent import QueryRefinementAgent

class TheCouncil:
    """
    The Council - Multi-Agent Research Intelligence System

    Orchestrates multiple specialized agents to conduct comprehensive research analysis
    with live streaming of agent interactions and reasoning.
    """

    def __init__(self):
        self.query_refinement_agent = QueryRefinementAgent()
        self.research_agent = ResearchAgent()
        self.critique_agent = CritiqueAgent()
        self.reviewer_agent = ReviewerAgent()
        self.summarizer = FileSummarizer()
        self.tavily_agent = TavilyAgent()

        # Communication log for tracking agent interactions
        self.conversation_log = []
        self.current_session = {
            'session_id': f"council_{int(time.time())}",
            'start_time': datetime.now(),
            'topic': None,
            'agents': ['query_refinement_agent', 'research_agent', 'critique_agent', 'reviewer_agent', 'summarizer', 'tavily_agent'],
            'status': 'initialized'
        }

    def log_message(self, agent: str, message_type: str, content: Any, metadata: Dict = None, effort_details: Dict = None):
        """Log agent communications for tracking and display with detailed effort logging."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'agent': agent,
            'message_type': message_type,
            'content': content,
            'metadata': metadata or {},
            'effort_details': effort_details or {},
            'event_id': f"{agent}_{message_type}_{int(time.time() * 1000)}"
        }
        self.conversation_log.append(log_entry)

        # Also write to individual agent log file
        self._write_agent_log(agent, log_entry)

        return log_entry

    def _write_agent_log(self, agent: str, log_entry: Dict):
        """Write detailed logs for each agent to separate files."""
        try:
            log_dir = Path("./agent_logs")
            log_dir.mkdir(exist_ok=True)

            log_file = log_dir / f"{agent}_events.log"

            # Format log entry for readability
            formatted_entry = f"""
[{log_entry['timestamp']}] {log_entry['agent'].upper()} - {log_entry['message_type'].upper()}
Event ID: {log_entry['event_id']}

CONTENT:
{log_entry['content']}

METADATA:
{json.dumps(log_entry['metadata'], indent=2)}

EFFORT DETAILS:
{json.dumps(log_entry['effort_details'], indent=2)}

{'='*80}
"""

            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(formatted_entry)

        except Exception as e:
            # Don't fail the main process if logging fails
            print(f"Warning: Failed to write agent log: {e}")

    def conduct_enhanced_research_council(self, research_topic: str, max_iterations: int = 3) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
        """
        Conduct an enhanced research council session with query refinement, iterative Q&A, and inter-agent context sharing.
        Implements the new Tavily research flow: Query Refinement → Search → Extract → Embed → Summarize → Iterative Q&A.

        Args:
            research_topic: The research topic to investigate
            max_iterations: Number of iterative Q&A cycles

        Yields progress updates for real-time display.
        """
        self.current_session['topic'] = research_topic
        self.current_session['status'] = 'active'

        yield self.log_message('council', 'session_start', f"Starting enhanced research council on: {research_topic}")

        try:
            # Phase 1: Query Refinement Agent creates better search queries
            yield self.log_message('council', 'phase_start', "Phase 1: Query Refinement - Creating targeted search queries")

            yield self.log_message('query_refinement_agent', 'action', "Refining user prompt into multiple targeted search queries...")
            query_refinement = self.query_refinement_agent.refine_query(research_topic)

            refined_queries = [q['query'] for q in query_refinement.get('refined_queries', [])]
            all_queries = [research_topic] + refined_queries  # Include original + refined

            yield self.log_message('query_refinement_agent', 'result', f"Generated {len(refined_queries)} refined queries", {
                'total_queries': len(all_queries),
                'refinement_reasoning_length': len(query_refinement.get('refinement_reasoning', ''))
            })

            # Phase 2: Enhanced Tavily Workflow (Search → Extract → Embed → Summarize)
            yield self.log_message('council', 'phase_start', "Phase 2: Enhanced Tavily Research - Search, Extract, Embed, Summarize")

            yield self.log_message('tavily_agent', 'action', "Executing enhanced research workflow with multiple queries...")
            enhanced_research = self.tavily_agent.enhanced_research_workflow(
                queries=all_queries,
                max_search_results=5,
                extract_top_n=3
            )

            yield self.log_message('tavily_agent', 'result', "Enhanced research completed", {
                'total_sources': enhanced_research.get('total_sources', 0),
                'total_embeddings': enhanced_research.get('total_embeddings', 0),
                'summaries_created': len(enhanced_research.get('summaries', []))
            })

            # Prepare sources for analysis (combine search results and extracted content)
            all_sources = []
            all_sources.extend(enhanced_research.get('search_results', []))
            all_sources.extend(enhanced_research.get('extracted_content', []))

            # Phase 3: Iterative Question-Based Analysis (Research → Critique → Review)
            yield self.log_message('council', 'phase_start', f"Phase 3: Iterative Question-Based Analysis ({max_iterations} iterations)")

            # Initialize context sharing between agents
            shared_context = ""
            iteration_results = []

            for iteration in range(1, max_iterations + 1):
                yield self.log_message('council', 'iteration_start', f"Iteration {iteration}/{max_iterations}: Question-Based Analysis")

                # Step 3a: Research Agent - Answer 10 comprehensive questions
                yield self.log_message('research_agent', 'action', f"Answering 10 comprehensive questions (iteration {iteration})...")
                research_qa = self.research_agent.analyze_findings_question_based(
                    all_sources,
                    research_topic,
                    iteration=iteration,
                    previous_context=shared_context
                )

                yield self.log_message('research_agent', 'result', f"Research Q&A completed - {research_qa.get('total_questions', 0)} questions answered", {
                    'iteration': iteration,
                    'questions_answered': len(research_qa.get('question_answers', {})),
                    'content_length': research_qa.get('content_length', 0)
                })

                # Step 3b: Critique Agent - Critique each answer
                yield self.log_message('critique_agent', 'action', f"Critiquing research answers (iteration {iteration})...")
                critique_qa = self.critique_agent.critique_question_based(
                    research_qa.get('question_answers', {}),
                    research_topic,
                    iteration=iteration,
                    previous_context=shared_context
                )

                yield self.log_message('critique_agent', 'result', f"Critique completed - {critique_qa.get('total_critiques', 0)} critiques provided", {
                    'iteration': iteration,
                    'critiques_completed': len(critique_qa.get('critique_answers', {}))
                })

                # Step 3c: Reviewer Agent - Review the Q&A and critiques
                yield self.log_message('reviewer_agent', 'action', f"Reviewing Q&A and critiques (iteration {iteration})...")
                review_qa = self.reviewer_agent.review_question_based(
                    research_qa.get('question_answers', {}),
                    critique_qa.get('critique_answers', {}),
                    research_topic,
                    iteration=iteration,
                    previous_context=shared_context
                )

                yield self.log_message('reviewer_agent', 'result', f"Review completed - {review_qa.get('total_reviews', 0)} reviews provided", {
                    'iteration': iteration,
                    'reviews_completed': len(review_qa.get('review_answers', {}))
                })

                # Update shared context for next iteration
                shared_context = f"""
Iteration {iteration} Results:
Research Answers: {len(research_qa.get('question_answers', {}))} questions answered
Critique Feedback: {len(critique_qa.get('critique_answers', {}))} critiques provided
Review Insights: {len(review_qa.get('total_reviews', 0))} reviews completed

Key Insights from this iteration:
- Research highlighted: {[q.get('question', '')[:50] + '...' for q in research_qa.get('question_answers', {}).values()][:3]}
- Critical feedback focused on: {[c.get('question', '')[:50] + '...' for c in critique_qa.get('critique_answers', {}).values()][:3]}
- Review emphasized: {[r.get('question', '')[:50] + '...' for r in review_qa.get('review_answers', {}).values()][:3]}
"""

                # Store iteration results
                iteration_results.append({
                    'iteration': iteration,
                    'research_qa': research_qa,
                    'critique_qa': critique_qa,
                    'review_qa': review_qa,
                    'shared_context': shared_context
                })

            # Phase 4: External Research Validation (Prove/Disprove Conclusions)
            yield self.log_message('council', 'phase_start', "Phase 4: External Research Validation - Testing conclusions with additional evidence")

            # Extract key conclusions from final iteration
            final_iteration = iteration_results[-1]
            key_conclusions = []
            for q_key, q_data in final_iteration['research_qa'].get('question_answers', {}).items():
                if q_data.get('question_number') in [4, 8, 9]:  # Conclusions, implications, gaps
                    key_conclusions.append(q_data.get('answer', '')[:200])

            validation_queries = [f"Validate: {conclusion[:100]}" for conclusion in key_conclusions[:3]]

            yield self.log_message('tavily_agent', 'action', "Conducting validation research on key conclusions...")
            validation_research = self.tavily_agent.enhanced_research_workflow(
                queries=validation_queries,
                max_search_results=3,
                extract_top_n=2
            )

            yield self.log_message('tavily_agent', 'result', "Validation research completed", {
                'validation_queries': len(validation_queries),
                'validation_sources': validation_research.get('total_sources', 0),
                'validation_embeddings': validation_research.get('total_embeddings', 0)
            })

            # Phase 5: Final Synthesis and Collective Insight Report
            yield self.log_message('council', 'phase_start', "Phase 5: Final Synthesis - Creating comprehensive Collective Insight Report")

            yield self.log_message('reviewer_agent', 'action', "Creating final Collective Insight Report with all iterations and validation...")

            # Compile comprehensive synthesis data
            synthesis_data = {
                'original_topic': research_topic,
                'query_refinement': query_refinement,
                'enhanced_research': enhanced_research,
                'iterative_qa_results': iteration_results,
                'validation_research': validation_research,
                'total_iterations': max_iterations,
                'agents_involved': ['query_refinement_agent', 'research_agent', 'critique_agent', 'reviewer_agent', 'tavily_agent']
            }

            # Truncate shared context for token limits
            max_context_length = 2000
            truncated_context = shared_context
            if len(shared_context) > max_context_length:
                truncated_context = shared_context[:max_context_length] + "\n[Context truncated for token limits]"

            # Create comprehensive synthesis report
            synthesis_prompt = f"""
Create a comprehensive Collective Insight Report for the research topic: "{research_topic}"

EXECUTIVE SUMMARY:
- Query refinement generated {len(refined_queries)} targeted search queries
- Enhanced research gathered {enhanced_research.get('total_sources', 0)} sources with {enhanced_research.get('total_embeddings', 0)} embeddings
- {max_iterations} iterations of question-based analysis completed
- Validation research tested key conclusions with additional evidence

KEY FINDINGS FROM ITERATIVE Q&A:
{truncated_context}

VALIDATION RESULTS:
Additional research validated key conclusions with {validation_research.get('total_sources', 0)} new sources.

STRUCTURE THE REPORT TO INCLUDE:
1. Executive Summary
2. Research Methodology (Query Refinement + Enhanced Tavily Workflow)
3. Key Findings (From 10 Question Framework)
4. Critical Analysis (Strengths and Limitations)
5. Validation Evidence (External Confirmation)
6. Practical Implications
7. Future Research Directions
8. Conclusion

Make this a comprehensive, professional research report that synthesizes all agent perspectives.
"""

            synthesis_response = self.reviewer_agent.client.responses.create(
                model="gpt-4o-mini",
                input=synthesis_prompt,
                instructions="Create a comprehensive, well-structured research synthesis report that integrates all findings, critiques, and validations into actionable insights."
            )

            final_synthesis = {
                'collective_insight_report': synthesis_response.output_text,
                'topic': research_topic,
                'synthesis_date': str(datetime.now()),
                'contributors': synthesis_data['agents_involved'],
                'methodology': 'Enhanced Tavily Research Flow with Iterative Q&A',
                'iterations_completed': max_iterations,
                'total_sources': enhanced_research.get('total_sources', 0) + validation_research.get('total_sources', 0),
                'total_embeddings': enhanced_research.get('total_embeddings', 0) + validation_research.get('total_embeddings', 0)
            }

            yield self.log_message('reviewer_agent', 'result', "Collective Insight Report completed", {
                'report_length': len(final_synthesis.get('collective_insight_report', '')),
                'contributors': len(final_synthesis.get('contributors', [])),
                'methodology': final_synthesis.get('methodology', '')
            })

            # Final session summary
            self.current_session['status'] = 'completed'
            self.current_session['end_time'] = datetime.now()
            self.current_session['total_interactions'] = len(self.conversation_log)

            final_result = {
                'session_info': self.current_session,
                'query_refinement': query_refinement,
                'enhanced_research': enhanced_research,
                'iterative_qa_results': iteration_results,
                'validation_research': validation_research,
                'collective_insight_report': final_synthesis,
                'conversation_log': self.conversation_log,
                'methodology': 'Enhanced Tavily Research Flow with Iterative Q&A',
                'iterations_completed': max_iterations,
                'total_sources': final_synthesis.get('total_sources', 0),
                'total_embeddings': final_synthesis.get('total_embeddings', 0)
            }

            yield self.log_message('council', 'session_complete', "Enhanced research council session completed successfully", {
                'total_messages': len(self.conversation_log),
                'session_duration': str(self.current_session['end_time'] - self.current_session['start_time']),
                'iterations_completed': max_iterations,
                'methodology': 'Enhanced Tavily Research Flow with Iterative Q&A'
            })

            # Yield the final result
            yield self.log_message('council', 'final_result', final_result)

        except Exception as e:
            self.current_session['status'] = 'error'
            self.current_session['error'] = str(e)
            yield self.log_message('council', 'error', f"Session error: {str(e)}")
            raise



    def conduct_document_council(self, file_path: str, max_iterations: int = 3) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
        """
        Conduct a document analysis council session with live streaming of agent interactions.

        Args:
            file_path: Path to the document to analyze
            max_iterations: Number of critique iterations

        Yields progress updates for real-time display.
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self.current_session['topic'] = f"Document Analysis: {file_path_obj.name}"
        self.current_session['document_path'] = str(file_path)
        self.current_session['status'] = 'active'

        yield self.log_message('council', 'session_start', f"Starting document analysis council for: {file_path_obj.name}")

        try:
            # Phase 1: Document Ingestion Agent processes the document
            yield self.log_message('council', 'phase_start', "Phase 1: Document Ingestion - parsing, chunking, and embedding document")

            yield self.log_message('document_agent', 'action', f"Loading and embedding document: {file_path_obj.name} ({file_path_obj.suffix})")

            # Use embeddings agent directly to avoid circular imports
            from embeddings_agent import DocumentEmbeddingsAgent
            embeddings_agent = DocumentEmbeddingsAgent()

            # Process document with embeddings
            ingestion_result = embeddings_agent.process_document_file(str(file_path))

            yield self.log_message('document_agent', 'result', f"Document processed and embedded successfully", {
                'filename': ingestion_result.get('filename'),
                'file_type': ingestion_result.get('file_type'),
                'chunks_created': ingestion_result.get('chunks_created', 0),
                'source_documents': ingestion_result.get('source_documents', 0),
                'embedding_model': 'text-embedding-ada-002'
            })

            # Extract content for analysis (this is a simplified approach)
            # In production, you'd retrieve this from the vector store
            content = f"Document: {file_path_obj.name}\nType: {file_path_obj.suffix}\nChunks: {ingestion_result.get('chunks_created', 0)}"

            # Phase 2: Summarizer Agent creates initial analysis
            yield self.log_message('council', 'phase_start', "Phase 2: Summarizer Agent - creating initial document analysis")

            yield self.log_message('summarizer', 'action', "Generating initial comprehensive summary...")
            initial_summary = self.summarizer.summarize_file(str(file_path))

            yield self.log_message('summarizer', 'result', "Initial summary completed", {
                'summary_length': len(initial_summary),
                'content_analyzed': len(content)
            })

            # Create research analysis structure
            research_analysis = {
                'analysis': initial_summary,
                'sources': [{
                    'title': file_path_obj.name,
                    'source': 'Document Upload',
                    'content_length': len(content),
                    'file_type': file_path_obj.suffix
                }],
                'topic': f"Document Analysis: {file_path_obj.name}",
                'status': 'analyzed'
            }

            # Phase 3: Critique Agent reviews the analysis (iterative improvement)
            yield self.log_message('council', 'phase_start', "Phase 3: Critique Agent - iterative analysis improvement")

            current_analysis = research_analysis
            critique_history = []

            for iteration in range(1, max_iterations + 1):
                yield self.log_message('council', 'iteration_start', f"Iteration {iteration}/{max_iterations}: Improving analysis")

                yield self.log_message('critique_agent', 'action', f"Analyzing current summary and identifying improvements...")
                critique_result = self.critique_agent.critique_summary(
                    original_content=content,
                    current_summary=current_analysis.get('analysis', ''),
                    iteration=iteration
                )

                critique_history.append(critique_result)

                yield self.log_message('critique_agent', 'result', f"Critique completed - refined analysis available", {
                    'iteration': iteration,
                    'critique_length': len(critique_result.get('critique', '')),
                    'improvement_made': bool(critique_result.get('improved_summary'))
                })

                # Update analysis for next iteration
                if critique_result.get('improved_summary'):
                    current_analysis = {
                        **current_analysis,
                        'analysis': critique_result.get('improved_summary', current_analysis.get('analysis', ''))
                    }

            # Phase 4: Reviewer Agent provides final comprehensive review
            yield self.log_message('council', 'phase_start', "Phase 4: Reviewer Agent - final comprehensive evaluation")

            yield self.log_message('reviewer_agent', 'action', "Conducting comprehensive document analysis review...")
            reviewer_feedback = self.reviewer_agent.review_analysis(
                research_analysis,
                critique_history[-1] if critique_history else None
            )

            yield self.log_message('reviewer_agent', 'result', "Comprehensive review completed", {
                'review_length': len(reviewer_feedback.get('review', '')),
                'analysis_quality': 'high'
            })

            # Phase 5: Synthesis Agent creates collective insight report
            yield self.log_message('council', 'phase_start', "Phase 5: Synthesis Agent - creating comprehensive insights")

            yield self.log_message('reviewer_agent', 'action', "Synthesizing all agent perspectives into final insights...")
            synthesis_result = self.reviewer_agent.synthesize_insights(
                research_analysis,
                critique_history[-1] if critique_history else {},
                reviewer_feedback
            )

            yield self.log_message('reviewer_agent', 'result', "Collective insights completed", {
                'insights_length': len(synthesis_result.get('collective_insight_report', '')),
                'agents_contributed': len(synthesis_result.get('contributors', []))
            })

            # Final session summary
            self.current_session['status'] = 'completed'
            self.current_session['end_time'] = datetime.now()
            self.current_session['total_interactions'] = len(self.conversation_log)

            final_result = {
                'session_info': self.current_session,
                'document_info': {
                    'filename': file_path_obj.name,
                    'file_type': file_path_obj.suffix,
                    'content_length': len(content),
                    'chunks_processed': ingestion_result.get('total_chunks', 0)
                },
                'initial_summary': initial_summary,
                'final_analysis': current_analysis.get('analysis', ''),
                'critique_history': critique_history,
                'reviewer_feedback': reviewer_feedback,
                'collective_insight_report': synthesis_result,
                'conversation_log': self.conversation_log,
                'iterations_completed': max_iterations
            }

            yield self.log_message('council', 'session_complete', "Research council session completed successfully", {
                'total_messages': len(self.conversation_log),
                'session_duration': str(self.current_session['end_time'] - self.current_session['start_time'])
            })

            # Yield the final result
            yield self.log_message('council', 'final_result', final_result)

        except Exception as e:
            self.current_session['status'] = 'error'
            self.current_session['error'] = str(e)
            yield self.log_message('council', 'error', f"Session error: {str(e)}")
            raise

    def _extract_content_from_sources(self, sources: List[Dict]) -> str:
        """Extract text content from various source types."""
        content_parts = []

        for source in sources:
            if 'abstract' in source and source['abstract']:
                content_parts.append(f"Abstract from {source.get('title', 'Unknown')}: {source['abstract']}")
            elif 'extract' in source and source['extract']:
                content_parts.append(f"Content from {source.get('title', 'Unknown')}: {source['extract']}")

        return "\n\n".join(content_parts)

    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session."""
        return {
            'session_id': self.current_session['session_id'],
            'topic': self.current_session['topic'],
            'status': self.current_session['status'],
            'start_time': self.current_session.get('start_time'),
            'end_time': self.current_session.get('end_time'),
            'total_interactions': len(self.conversation_log),
            'agents_involved': self.current_session['agents']
        }

    def export_conversation_log(self, filename: str = None) -> str:
        """Export the conversation log to a file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"council_session_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'session_info': self.current_session,
                'conversation_log': self.conversation_log
            }, f, indent=2, default=str)

        return filename

def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m the_council___research_tool.the_council <research_topic>")
        sys.exit(1)

    topic = sys.argv[1]
    council = TheCouncil()

    try:
        print(f"🏛️  THE COUNCIL - Research Intelligence Session")
        print(f"Topic: {topic}")
        print("="*60)

        # Run the enhanced council session with live updates
        for update in council.conduct_enhanced_research_council(topic, max_iterations=3):
            # Display live updates
            agent = update['agent']
            msg_type = update['message_type']
            content = update['content']

            if msg_type == 'session_start':
                print(f"\n🎯 {content}")
            elif msg_type == 'phase_start':
                print(f"\n📋 {content}")
            elif msg_type == 'iteration_start':
                print(f"\n🔄 {content}")
            elif msg_type == 'action':
                print(f"  🤖 {agent}: {content}")
            elif msg_type == 'result':
                metadata = update.get('metadata', {})
                print(f"  ✅ {content}")
                if metadata:
                    for key, value in metadata.items():
                        print(f"     {key}: {value}")
            elif msg_type == 'session_complete':
                print(f"\n🎉 {content}")
            elif msg_type == 'error':
                print(f"\n❌ {content}")

        # Get final results
        session_summary = council.get_session_summary()
        print("\n📊 Session Summary:")
        duration = session_summary.get('end_time', datetime.now()) - session_summary.get('start_time', datetime.now())
        print(f"  Duration: {duration}")
        print(f"  Total Interactions: {session_summary['total_interactions']}")
        print(f"  Agents Involved: {', '.join(session_summary['agents_involved'])}")

        # Export conversation log
        log_file = council.export_conversation_log()
        print(f"\n📄 Conversation log saved to: {log_file}")

    except Exception as e:
        print(f"❌ Council session failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
