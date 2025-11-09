import os
from pathlib import Path
from typing import Dict, Any
from openai import AzureOpenAI

class CritiqueAgent:
    def __init__(self):
        self.client = AzureOpenAI(
            api_version="2025-03-01-preview",
            azure_endpoint="https://nachalo-2324-resource.cognitiveservices.azure.com/",
            api_key="os.getenv("AZURE_OPENAI_API_KEY", "your-api-key-here")",
        )

    def critique_question_based(self, question_answers: Dict, research_topic: str, iteration: int = 1, previous_context: str = "") -> Dict[str, Any]:
        """
        Critique question-based analysis using the 10 comprehensive questions.

        Args:
            question_answers: Dict of question answers from research agent
            research_topic: The research topic
            iteration: Current iteration number
            previous_context: Context from previous iterations

        Returns:
            Question-based critique results
        """

        # 10 comprehensive questions for critique
        critique_questions = [
            "How comprehensive and accurate is the assessment of information content and source quality?",
            "Are the key takeaways and findings properly identified and substantiated?",
            "Is the analysis of methods and approaches thorough and critical?",
            "Do the conclusions and implications logically follow from the evidence?",
            "Is the supporting evidence adequately evaluated for strength and relevance?",
            "Are limitations and potential biases clearly identified and addressed?",
            "How well does the analysis connect to existing knowledge and literature?",
            "Are practical applications and real-world implications realistic and well-supported?",
            "Are research gaps and unanswered questions clearly articulated?",
            "Are suggested future research directions feasible and well-justified?"
        ]

        critique_answers = {}

        # Answer each critique question
        for i, question in enumerate(critique_questions, 1):
            # Get corresponding research answer
            research_key = f"question_{i}"
            research_answer = question_answers.get(research_key, {}).get('answer', 'No answer provided')

            # Truncate research answer to stay within token limits
            max_answer_length = 2000  # Limit research answer length
            if len(research_answer) > max_answer_length:
                research_answer = research_answer[:max_answer_length] + "\n[Answer truncated for token limits]"

            # Also truncate previous context
            max_context_length = 1000
            truncated_context = previous_context
            if len(previous_context) > max_context_length:
                truncated_context = previous_context[:max_context_length] + "\n[Context truncated for token limits]"

            prompt = f"""
You are a Critique Agent in iteration {iteration}. Critically evaluate the following research analysis answer.

CRITIQUE QUESTION {i}: {question}

ORIGINAL RESEARCH ANSWER:
{research_answer}

RESEARCH TOPIC: {research_topic}

PREVIOUS CONTEXT FROM OTHER AGENTS:
{truncated_context}

YOUR TASK:
Provide a detailed critique of this answer. Evaluate its strengths and weaknesses, suggest improvements, and provide a revised version if needed. Consider:

1. **Accuracy**: Is the information correct and well-supported?
2. **Completeness**: Does it address all aspects of the question?
3. **Logic**: Is the reasoning sound and well-structured?
4. **Evidence**: Is adequate evidence provided?
5. **Objectivity**: Is the analysis balanced and unbiased?
6. **Clarity**: Is the answer clear and well-written?

Provide constructive feedback and, if appropriate, suggest a revised answer.
"""

            try:
                response = self.client.responses.create(
                    model="gpt-4o-mini",
                    input=prompt,
                    instructions=f"You are critiquing answer {i}. Be thorough, constructive, and evidence-based."
                )

                critique_answers[f"critique_{i}"] = {
                    'question': question,
                    'critique': response.output_text.strip(),
                    'original_answer': research_answer,
                    'question_number': i
                }
            except Exception as e:
                critique_answers[f"critique_{i}"] = {
                    'question': question,
                    'critique': f"Error generating critique: {str(e)}",
                    'original_answer': research_answer,
                    'question_number': i
                }

        return {
            'critique_answers': critique_answers,
            'topic': research_topic,
            'iteration': iteration,
            'total_critiques': len(critique_questions),
            'status': 'critiqued'
        }

    def critique_summary(self, original_content: str, current_summary: str, iteration: int = 1) -> Dict[str, Any]:
        """Legacy method for backward compatibility."""
        # Convert to question-based format for legacy support
        mock_answers = {
            'question_1': {'answer': current_summary}
        }
        return self.critique_question_based(mock_answers, "Legacy Summary", iteration)

    def evaluate_final_summary(self, original_content: str, final_summary: str) -> Dict[str, Any]:
        """
        Provide final evaluation of the summary after all iterations.
        """
        prompt = f"""
Evaluate the final summary after 3 iterations of critique and improvement.

**ORIGINAL DOCUMENT:**
{original_content[:2000]}... (truncated for analysis)

**FINAL SUMMARY:**
{final_summary}

**EVALUATION CRITERIA:**
- **Completeness** (1-10): How well does it capture all key information?
- **Accuracy** (1-10): How accurately does it represent the original content?
- **Clarity** (1-10): How clear and well-structured is the summary?
- **Conciseness** (1-10): How efficiently is information conveyed?
- **Overall Quality** (1-10): Comprehensive assessment

Provide scores and detailed justification for each criterion.
"""

        response = self.client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            instructions="You are a summary evaluation expert. Provide detailed, numerical scores with specific justifications."
        )

        return {
            'evaluation': response.output_text,
            'final_summary': final_summary,
            'status': 'evaluated'
        }

def main():
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m the_council___research_tool.critique_agent <document_file> <summary_file>")
        sys.exit(1)

    document_file = sys.argv[1]
    summary_file = sys.argv[2]

    # Read files
    with open(document_file, 'r', encoding='utf-8') as f:
        original_content = f.read()

    with open(summary_file, 'r', encoding='utf-8') as f:
        current_summary = f.read()

    agent = CritiqueAgent()

    try:
        result = agent.critique_summary(original_content, current_summary, iteration=1)
        print("Critique completed!")
        print("=" * 50)
        print(result['critique'])
        print("=" * 50)
        print("IMPROVED SUMMARY:")
        print(result['improved_summary'])
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
