import os
from typing import Dict, Any, List
from openai import AzureOpenAI

class QueryRefinementAgent:
    """
    Query Refinement Agent for creating better search queries.
    Refines user prompts into multiple targeted search queries.
    """

    def __init__(self):
        self.client = AzureOpenAI(
            api_version="2025-03-01-preview",
            azure_endpoint="https://nachalo-2324-resource.cognitiveservices.azure.com/",
            api_key="os.getenv("AZURE_OPENAI_API_KEY", "your-api-key-here")",
        )

    def refine_query(self, user_prompt: str, context: str = "") -> Dict[str, Any]:
        """
        Refine user prompt into multiple targeted search queries with detailed effort logging.

        Args:
            user_prompt: Original user research prompt
            context: Additional context from previous research

        Returns:
            Dict containing refined queries and reasoning
        """
        # Log effort: Starting query refinement
        effort_details = {
            'stage': 'analysis',
            'task': 'analyzing_user_prompt',
            'input_length': len(user_prompt),
            'context_length': len(context),
            'timestamp': 'preprocessing'
        }

        prompt = f"""
You are a Query Refinement Specialist. Your task is to transform a user's research prompt into multiple targeted, effective search queries.

EFFORT TRACKING: Log your thought process and analysis steps.

USER RESEARCH PROMPT:
{user_prompt}

ADDITIONAL CONTEXT:
{context}

YOUR TASK:
Create 3-5 highly targeted search queries that will yield the most relevant and comprehensive results. Each query should:

1. **Be Specific**: Focus on key concepts and avoid vague terms
2. **Include Variations**: Use different phrasings and synonyms
3. **Target Different Aspects**: Cover various angles of the research topic
4. **Be Search-Engine Friendly**: Use natural language that search engines understand
5. **Include Time Indicators**: Add temporal context when relevant (recent developments, historical context, etc.)

For each query, provide:
- The query text
- Rationale for why this query will be effective
- What type of information it should return

Format your response as a structured list of queries with explanations.

LOG YOUR EFFORTS: Document your analysis approach and reasoning process.
"""

        # Log effort: Sending to AI
        effort_details.update({
            'stage': 'ai_processing',
            'task': 'generating_queries',
            'prompt_length': len(prompt),
            'timestamp': 'ai_request'
        })

        response = self.client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            instructions="You are an expert at crafting effective search queries. Provide specific, actionable queries with clear reasoning. Include your analytical process in the response."
        )

        # Log effort: Processing response
        effort_details.update({
            'stage': 'postprocessing',
            'task': 'parsing_response',
            'response_length': len(response.output_text),
            'timestamp': 'response_received'
        })

        # Parse the response to extract queries
        response_text = response.output_text

        # Simple parsing - look for numbered queries
        queries = []
        lines = response_text.split('\n')

        current_query = None
        current_rationale = None

        for line in lines:
            line = line.strip()
            if line.startswith(('1.', '2.', '3.', '4.', '5.')) and 'Query:' in line:
                if current_query:
                    queries.append({
                        'query': current_query,
                        'rationale': current_rationale or 'Generated refined query'
                    })
                # Extract query text
                query_part = line.split('Query:', 1)[1].strip() if 'Query:' in line else line.split('.', 1)[1].strip()
                current_query = query_part
                current_rationale = None
            elif current_query and ('Rationale:' in line or 'Why:' in line or 'Because:' in line):
                rationale_part = line.split(':', 1)[1].strip() if ':' in line else line
                current_rationale = rationale_part

        # Add the last query
        if current_query:
            queries.append({
                'query': current_query,
                'rationale': current_rationale or 'Generated refined query'
            })

        # Fallback: if parsing failed, create basic queries
        if not queries:
            base_queries = [
                user_prompt,
                f"latest research on {user_prompt}",
                f"{user_prompt} analysis and findings",
                f"{user_prompt} methods and approaches"
            ]
            queries = [{'query': q, 'rationale': 'Basic refined query'} for q in base_queries[:4]]

        # Log final effort details
        effort_details.update({
            'stage': 'completed',
            'task': 'finalizing_results',
            'queries_generated': len(queries),
            'parsing_successful': len(queries) > 0,
            'timestamp': 'completed'
        })

        return {
            'original_prompt': user_prompt,
            'refined_queries': queries,
            'total_queries': len(queries),
            'refinement_reasoning': response_text,
            'effort_log': effort_details
        }

    def combine_with_context(self, refined_queries: List[str], user_prompt: str) -> List[str]:
        """
        Combine refined queries with original user prompt for comprehensive search.

        Args:
            refined_queries: List of refined query strings
            user_prompt: Original user prompt

        Returns:
            Combined list of search queries
        """
        combined_queries = [user_prompt]  # Start with original

        # Add refined queries
        for query in refined_queries:
            if query not in combined_queries:
                combined_queries.append(query)

        return combined_queries

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m the_council___research_tool.query_refinement_agent <user_prompt>")
        sys.exit(1)

    user_prompt = sys.argv[1]
    agent = QueryRefinementAgent()

    try:
        print(f"🔍 Refining query: {user_prompt}")
        result = agent.refine_query(user_prompt)

        print(f"\n✅ Generated {result['total_queries']} refined queries:")
        for i, query_info in enumerate(result['refined_queries'], 1):
            print(f"\n{i}. {query_info['query']}")
            print(f"   Rationale: {query_info['rationale']}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
