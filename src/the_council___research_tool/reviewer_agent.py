import os
from pathlib import Path
from typing import Dict, Any, List
from openai import AzureOpenAI
import json
from datetime import datetime

class ReviewerAgent:
    def __init__(self):
        self.client = AzureOpenAI(
            api_version="2025-03-01-preview",
            azure_endpoint="https://nachalo-2324-resource.cognitiveservices.azure.com/",
            api_key="os.getenv("AZURE_OPENAI_API_KEY", "your-api-key-here")",
        )

    def review_question_based(self, question_answers: Dict, critique_answers: Dict, research_topic: str, iteration: int = 1, previous_context: str = "") -> Dict[str, Any]:
        """
        Review question-based analysis and critiques using the 10 comprehensive questions.

        Args:
            question_answers: Dict of research question answers
            critique_answers: Dict of critique answers
            research_topic: The research topic
            iteration: Current iteration number
            previous_context: Context from previous iterations

        Returns:
            Question-based review results
        """

        # 10 comprehensive questions for review
        review_questions = [
            "Does the overall analysis demonstrate methodological rigor and scholarly quality?",
            "How well are key findings supported by evidence across all answers?",
            "Is there logical consistency and coherence across the 10 question answers?",
            "How effectively are potential biases and limitations addressed?",
            "Does the analysis achieve comprehensive coverage of the research topic?",
            "Are practical implications and applications well-developed and realistic?",
            "How do the research findings contribute to existing knowledge?",
            "Are research gaps and future directions clearly identified and prioritized?",
            "How well do the critiques improve upon the original analysis?",
            "What is the overall quality and scholarly contribution of this research?"
        ]

        review_answers = {}

        # Review each question-answer-critique triad
        for i, question in enumerate(review_questions, 1):
            research_key = f"question_{i}"
            critique_key = f"critique_{i}"

            research_answer = question_answers.get(research_key, {}).get('answer', 'No research answer provided')
            critique_answer = critique_answers.get(critique_key, {}).get('critique', 'No critique provided')

            # Truncate inputs to stay within token limits
            max_answer_length = 1500  # Limit research answer length
            if len(research_answer) > max_answer_length:
                research_answer = research_answer[:max_answer_length] + "\n[Research answer truncated for token limits]"

            max_critique_length = 1000  # Limit critique length
            if len(critique_answer) > max_critique_length:
                critique_answer = critique_answer[:max_critique_length] + "\n[Critique truncated for token limits]"

            max_context_length = 800  # Limit previous context
            truncated_context = previous_context
            if len(previous_context) > max_context_length:
                truncated_context = previous_context[:max_context_length] + "\n[Context truncated for token limits]"

            prompt = f"""
You are a Senior Research Reviewer Agent in iteration {iteration}. Review the following research question, answer, and critique.

REVIEW QUESTION {i}: {question}

ORIGINAL RESEARCH ANSWER:
{research_answer}

CRITIQUE OF RESEARCH ANSWER:
{critique_answer}

RESEARCH TOPIC: {research_topic}

PREVIOUS CONTEXT FROM OTHER AGENTS:
{truncated_context}

YOUR TASK:
Provide a comprehensive review that evaluates the quality, accuracy, and scholarly merit of the research answer and its critique. Consider:

1. **Scholarly Quality**: Academic rigor and methodological soundness
2. **Evidence-Based Analysis**: How well claims are supported by evidence
3. **Critical Thinking**: Depth of analysis and identification of complexities
4. **Practical Relevance**: Real-world implications and applications
5. **Original Contribution**: What new insights or perspectives are offered
6. **Areas for Improvement**: Specific recommendations for enhancement

Provide balanced, evidence-based feedback that acknowledges strengths while identifying areas for improvement.
"""

            try:
                response = self.client.responses.create(
                    model="gpt-4o-mini",
                    input=prompt,
                    instructions=f"You are reviewing question {i}. Be thorough, scholarly, and constructive."
                )

                review_answers[f"review_{i}"] = {
                    'question': question,
                    'review': response.output_text.strip(),
                    'research_answer': research_answer,
                    'critique_answer': critique_answer,
                    'question_number': i
                }
            except Exception as e:
                review_answers[f"review_{i}"] = {
                    'question': question,
                    'review': f"Error generating review: {str(e)}",
                    'research_answer': research_answer,
                    'critique_answer': critique_answer,
                    'question_number': i
                }

        return {
            'review_answers': review_answers,
            'topic': research_topic,
            'iteration': iteration,
            'total_reviews': len(review_questions),
            'status': 'reviewed'
        }

    def review_analysis(self, research_analysis: Dict, critique_feedback: Dict = None) -> Dict[str, Any]:
        """Legacy method for backward compatibility."""
        # Convert to question-based format for legacy support
        mock_answers = {
            'question_1': {'answer': research_analysis.get('analysis', '')}
        }
        mock_critiques = {
            'critique_1': {'critique': critique_feedback.get('critique', '') if critique_feedback else ''}
        }
        return self.review_question_based(mock_answers, mock_critiques, research_analysis.get('topic', 'Unknown'))

    def synthesize_insights(self, research_analysis: Dict, critique_feedback: Dict,
                           reviewer_feedback: Dict) -> Dict[str, Any]:
        """Synthesize all inputs into collective insights."""

        topic = research_analysis.get('topic', 'Unknown')

        synthesis_prompt = f"""
You are a Synthesis Agent responsible for creating a "Collective Insight Report" from multiple research perspectives.

RESEARCH TOPIC: {topic}

ORIGINAL RESEARCH ANALYSIS:
{research_analysis.get('analysis', '')}

CRITIQUE FEEDBACK:
{critique_feedback.get('critique', '') if critique_feedback else 'No critique available'}

REVIEWER FEEDBACK:
{reviewer_feedback.get('review', '')}

YOUR TASK:
Create a comprehensive "Collective Insight Report" that includes:

1. **Core Consensus**: What all perspectives agree upon
2. **Key Insights**: Most important findings and implications
3. **Resolved Conflicts**: How differing views were reconciled
4. **Remaining Uncertainties**: Areas still needing clarification
5. **Actionable Recommendations**: Concrete next steps
6. **Hypothesis Generation**: New hypotheses worth exploring
7. **Research Gaps**: Critical areas needing further investigation

Structure your report professionally with clear sections, citations to sources, and reasoning traces.
Include confidence levels for different conclusions and identify assumptions made.
"""

        response = self.client.responses.create(
            model="gpt-4o-mini",
            input=synthesis_prompt,
            instructions="You are an expert synthesizer. Create a comprehensive, well-structured report that integrates multiple perspectives into actionable insights."
        )

        synthesis = response.output_text

        return {
            'collective_insight_report': synthesis,
            'topic': topic,
            'synthesis_date': str(datetime.now()),
            'contributors': ['research_agent', 'critique_agent', 'reviewer_agent'],
            'status': 'synthesized'
        }

def main():
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python -m the_council___research_tool.reviewer_agent <research_topic>")
        sys.exit(1)

    topic = sys.argv[1]
    agent = ReviewerAgent()

    try:
        print(f"🔍 Reviewing research on: {topic}")

        # For testing, create mock data
        mock_research = {
            'analysis': f"This is a comprehensive analysis of {topic} covering various aspects and findings from multiple sources.",
            'sources': [{'title': 'Sample Paper 1', 'source': 'arXiv'}, {'title': 'Wikipedia Article', 'source': 'Wikipedia'}],
            'topic': topic
        }

        mock_critique = {
            'critique': f"Critique feedback on the {topic} analysis, identifying strengths and areas for improvement."
        }

        # Perform review
        print("📝 Conducting review...")
        review = agent.review_analysis(mock_research, mock_critique)

        # Perform synthesis
        print("🔄 Synthesizing insights...")
        synthesis = agent.synthesize_insights(mock_research, mock_critique, review)

        print("\n" + "="*60)
        print("REVIEW:")
        print("="*60)
        print(review['review'])

        print("\n" + "="*60)
        print("COLLECTIVE INSIGHT REPORT:")
        print("="*60)
        print(synthesis['collective_insight_report'])

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
