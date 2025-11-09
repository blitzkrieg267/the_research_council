import os
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
from openai import AzureOpenAI


class TavilyAgent:
    """
    Tavily Agent for web search, content extraction, and crawling.
    Integrates with Tavily API for comprehensive web research capabilities.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Tavily Agent.

        Args:
            api_key: Tavily API key. If not provided, will check environment variable.
        """
        self.api_key = api_key or os.getenv('TAVILY_API_KEY', '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')
        self.base_url = "https://api.tavily.com"

        # Initialize Azure OpenAI for embeddings and summarization
        self.client = AzureOpenAI(
            api_version="2025-03-01-preview",
            azure_endpoint="https://nachalo-2324-resource.cognitiveservices.azure.com/",
            api_key="os.getenv("AZURE_OPENAI_API_KEY", "your-api-key-here")",
        )

    def search(self,
               query: str,
               max_results: int = 5,
               search_depth: str = "advanced",
               include_answer: bool = True,
               include_raw_content: bool = False,
               topic: str = "general",
               time_range: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform a web search using Tavily Search API.

        Args:
            query: Search query
            max_results: Maximum number of results (1-20)
            search_depth: "basic" or "advanced"
            include_answer: Include AI-generated answer
            include_raw_content: Include raw content from sources
            topic: "general" or "news"
            time_range: Time filter ("day", "week", "month", "year")

        Returns:
            Search results with formatted data
        """
        try:
            url = f"{self.base_url}/search"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
                "include_answer": include_answer,
                "include_raw_content": include_raw_content,
                "topic": topic
            }

            if time_range:
                payload["time_range"] = time_range

            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Format results for our system
            formatted_results = {
                'query': data.get('query', query),
                'answer': data.get('answer', ''),
                'results': [],
                'response_time': data.get('response_time', 0),
                'total_results': len(data.get('results', []))
            }

            for result in data.get('results', []):
                formatted_results['results'].append({
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'content': result.get('content', ''),
                    'source': 'tavily_search',
                    'score': result.get('score', 0),
                    'published_date': result.get('published_date'),
                    'search_depth': search_depth
                })

            return formatted_results

        except Exception as e:
            return {
                'error': str(e),
                'query': query,
                'results': [],
                'total_results': 0
            }

    def extract(self,
                urls: List[str],
                extract_depth: str = "advanced",
                include_images: bool = False,
                format_type: str = "markdown") -> Dict[str, Any]:
        """
        Extract content from URLs using Tavily Extract API.

        Args:
            urls: List of URLs to extract content from
            extract_depth: "basic" or "advanced"
            include_images: Include images in results
            format_type: "markdown" or "text"

        Returns:
            Extracted content from URLs
        """
        try:
            url = f"{self.base_url}/extract"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "urls": urls,
                "extract_depth": extract_depth,
                "include_images": include_images,
                "format": format_type
            }

            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()

            data = response.json()

            # Format extracted results
            formatted_results = {
                'extracted_content': [],
                'failed_urls': data.get('failed_results', []),
                'response_time': data.get('response_time', 0),
                'total_extracted': len(data.get('results', []))
            }

            for result in data.get('results', []):
                formatted_results['extracted_content'].append({
                    'url': result.get('url', ''),
                    'title': f"Extracted from {result.get('url', '')}",
                    'content': result.get('raw_content', ''),
                    'source': 'tavily_extract',
                    'extract_depth': extract_depth,
                    'format': format_type
                })

            return formatted_results

        except Exception as e:
            return {
                'error': str(e),
                'extracted_content': [],
                'failed_urls': urls,
                'total_extracted': 0
            }

    def crawl(self,
              base_url: str,
              instructions: Optional[str] = None,
              max_depth: int = 1,
              max_breadth: int = 20,
              limit: int = 50,
              extract_depth: str = "basic") -> Dict[str, Any]:
        """
        Crawl a website using Tavily Crawl API.

        Args:
            base_url: Root URL to start crawling
            instructions: Natural language instructions for crawler
            max_depth: Maximum crawl depth
            max_breadth: Maximum links per level
            limit: Total pages to process
            extract_depth: Content extraction depth

        Returns:
            Crawled website content
        """
        try:
            url = f"{self.base_url}/crawl"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "url": base_url,
                "max_depth": max_depth,
                "max_breadth": max_breadth,
                "limit": limit,
                "extract_depth": extract_depth
            }

            if instructions:
                payload["instructions"] = instructions

            response = requests.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()

            data = response.json()

            # Format crawl results
            formatted_results = {
                'base_url': data.get('base_url', base_url),
                'crawled_pages': [],
                'response_time': data.get('response_time', 0),
                'total_pages': len(data.get('results', []))
            }

            for result in data.get('results', []):
                formatted_results['crawled_pages'].append({
                    'url': result.get('url', ''),
                    'title': f"Crawled: {result.get('url', '')}",
                    'content': result.get('raw_content', ''),
                    'source': 'tavily_crawl',
                    'crawl_instructions': instructions
                })

            return formatted_results

        except Exception as e:
            return {
                'error': str(e),
                'base_url': base_url,
                'crawled_pages': [],
                'total_pages': 0
            }

    def comprehensive_research(self,
                              query: str,
                              max_search_results: int = 5,
                              extract_top_n: int = 3,
                              include_crawl: bool = False,
                              crawl_instructions: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive research using multiple Tavily APIs.

        Args:
            query: Research query
            max_search_results: Maximum search results
            extract_top_n: Number of top results to extract
            include_crawl: Whether to include crawling
            crawl_instructions: Instructions for crawler

        Returns:
            Comprehensive research results
        """
        research_results = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'search_results': [],
            'extracted_content': [],
            'crawled_content': [],
            'total_sources': 0
        }

        # Step 1: Search
        search_data = self.search(query, max_results=max_search_results)
        research_results['search_results'] = search_data.get('results', [])

        # Step 2: Extract content from top results
        if research_results['search_results']:
            top_urls = [result['url'] for result in research_results['search_results'][:extract_top_n]]
            extract_data = self.extract(top_urls)
            research_results['extracted_content'] = extract_data.get('extracted_content', [])

        # Step 3: Optional crawling (if requested and we have results)
        if include_crawl and research_results['search_results']:
            # Try to crawl the first domain
            first_url = research_results['search_results'][0]['url']
            # Extract domain for crawling
            try:
                from urllib.parse import urlparse
                domain = urlparse(first_url).netloc
                crawl_url = f"https://{domain}"

                crawl_data = self.crawl(
                    crawl_url,
                    instructions=crawl_instructions,
                    max_depth=2,
                    limit=20
                )
                research_results['crawled_content'] = crawl_data.get('crawled_pages', [])
            except:
                research_results['crawled_content'] = []

        # Calculate totals
        research_results['total_sources'] = (
            len(research_results['search_results']) +
            len(research_results['extracted_content']) +
            len(research_results['crawled_content'])
        )

        return research_results

    def embed_content(self, content_list: List[str]) -> List[List[float]]:
        """
        Create embeddings for content using Azure OpenAI.
        Truncates content to stay within token limits.

        Args:
            content_list: List of text content to embed

        Returns:
            List of embedding vectors
        """
        try:
            embeddings = []

            # Truncate content to stay within token limits (roughly 4 chars per token)
            # text-embedding-ada-002 has 8192 token limit, so limit to ~3000 chars per piece
            max_chars_per_content = 3000

            # Process in smaller batches to avoid token limits
            batch_size = 5  # Reduced batch size

            for i in range(0, len(content_list), batch_size):
                batch_texts = content_list[i:i + batch_size]

                # Truncate each piece of content
                truncated_batch = []
                for content in batch_texts:
                    if len(content) > max_chars_per_content:
                        # Truncate at word boundary if possible
                        truncated = content[:max_chars_per_content]
                        last_space = truncated.rfind(' ')
                        if last_space > max_chars_per_content * 0.8:  # Only if space is reasonably close
                            truncated = truncated[:last_space]
                        truncated_batch.append(truncated)
                    else:
                        truncated_batch.append(content)

                response = self.client.embeddings.create(
                    input=truncated_batch,
                    model="text-embedding-ada-002"
                )

                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)

            return embeddings

        except Exception as e:
            print(f"Error creating embeddings: {e}")
            return []

    def summarize_content(self, content: str, max_length: int = 500) -> str:
        """
        Summarize content using Azure OpenAI.
        Truncates input content to stay within token limits.

        Args:
            content: Text content to summarize
            max_length: Maximum length of summary

        Returns:
            Summarized content
        """
        try:
            # Truncate content more aggressively to stay within token limits
            # gpt-4o-mini has token limits, so limit input content significantly
            max_input_chars = 2000  # Reduced from 4000

            if len(content) > max_input_chars:
                # Truncate at word boundary if possible
                truncated_content = content[:max_input_chars]
                last_space = truncated_content.rfind(' ')
                if last_space > max_input_chars * 0.8:
                    truncated_content = truncated_content[:last_space]
            else:
                truncated_content = content

            prompt = f"""
Summarize the following content in a clear, concise manner. Focus on the key information, findings, and insights.

CONTENT:
{truncated_content}

SUMMARY REQUIREMENTS:
- Keep it under {max_length} characters
- Include main points and conclusions
- Maintain factual accuracy
- Use clear, professional language

SUMMARY:
"""

            response = self.client.responses.create(
                model="gpt-4o-mini",
                input=prompt,
                instructions="You are a content summarizer. Create clear, accurate summaries that capture the essence of the content."
            )

            return response.output_text.strip()

        except Exception as e:
            print(f"Error summarizing content: {e}")
            return content[:200] + "..." if len(content) > 200 else content

    def enhanced_research_workflow(self,
                                  queries: List[str],
                                  max_search_results: int = 5,
                                  extract_top_n: int = 3) -> Dict[str, Any]:
        """
        Enhanced research workflow: Search → Extract URLs → Embed content → Summarize → Add to context.
        Includes detailed effort logging for each step.

        Args:
            queries: List of search queries
            max_search_results: Maximum search results per query
            extract_top_n: Number of top results to extract per query

        Returns:
            Enhanced research results with embeddings and summaries
        """
        effort_log = {
            'workflow_start': datetime.now().isoformat(),
            'total_queries': len(queries),
            'stages': []
        }

        enhanced_results = {
            'queries': queries,
            'timestamp': datetime.now().isoformat(),
            'search_results': [],
            'extracted_content': [],
            'embeddings': [],
            'summaries': [],
            'total_sources': 0,
            'total_embeddings': 0,
            'effort_details': effort_log
        }

        all_content_to_embed = []

        # Stage 1: Web Search
        search_start = datetime.now()
        effort_log['stages'].append({
            'stage': 'web_search',
            'start_time': search_start.isoformat(),
            'queries_to_process': len(queries)
        })

        # Process each query
        for i, query in enumerate(queries):
            query_effort = {
                'query_index': i,
                'query_text': query,
                'search_results_found': 0,
                'extraction_attempts': 0,
                'content_extracted': 0
            }

            # Step 1: Search
            search_data = self.search(query, max_results=max_search_results)
            results = search_data.get('results', [])
            enhanced_results['search_results'].extend(results)
            query_effort['search_results_found'] = len(results)

            # Step 2: Extract URLs from search results
            if results:
                top_urls = [result['url'] for result in results[:extract_top_n]]
                extract_data = self.extract(top_urls)
                extracted_items = extract_data.get('extracted_content', [])
                enhanced_results['extracted_content'].extend(extracted_items)

                query_effort['extraction_attempts'] = len(top_urls)
                query_effort['content_extracted'] = len(extracted_items)

                # Prepare content for embedding and summarization
                for item in extracted_items:
                    content = item.get('content', '')
                    if content:
                        all_content_to_embed.append(content)

            effort_log['stages'][-1]['queries_processed'] = i + 1
            effort_log['stages'][-1]['query_details'] = effort_log['stages'][-1].get('query_details', []) + [query_effort]

        search_end = datetime.now()
        effort_log['stages'][-1]['end_time'] = search_end.isoformat()
        effort_log['stages'][-1]['duration_seconds'] = (search_end - search_start).total_seconds()

        # Stage 2: Content Processing (Embeddings & Summaries)
        processing_start = datetime.now()
        effort_log['stages'].append({
            'stage': 'content_processing',
            'start_time': processing_start.isoformat(),
            'content_items': len(all_content_to_embed),
            'embedding_batches': 0,
            'summaries_created': 0
        })

        # Step 3: Create embeddings for all extracted content
        if all_content_to_embed:
            embeddings = self.embed_content(all_content_to_embed)
            enhanced_results['embeddings'] = embeddings
            enhanced_results['total_embeddings'] = len(embeddings)
            effort_log['stages'][-1]['embedding_batches'] = len(embeddings) // 5 + 1  # Approximate batches

        # Step 4: Summarize content
        for i, content in enumerate(all_content_to_embed):
            summary = self.summarize_content(content)
            enhanced_results['summaries'].append({
                'content_index': i,
                'summary': summary,
                'original_length': len(content)
            })

        effort_log['stages'][-1]['summaries_created'] = len(all_content_to_embed)

        processing_end = datetime.now()
        effort_log['stages'][-1]['end_time'] = processing_end.isoformat()
        effort_log['stages'][-1]['duration_seconds'] = (processing_end - processing_start).total_seconds()

        # Calculate totals
        enhanced_results['total_sources'] = (
            len(enhanced_results['search_results']) +
            len(enhanced_results['extracted_content'])
        )

        # Final effort summary
        effort_log['workflow_end'] = datetime.now().isoformat()
        effort_log['total_duration_seconds'] = (datetime.now() - datetime.fromisoformat(effort_log['workflow_start'])).total_seconds()
        effort_log['total_sources_found'] = enhanced_results['total_sources']
        effort_log['total_embeddings_created'] = enhanced_results['total_embeddings']

        return enhanced_results


def main():
    """Test the Tavily Agent."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m the_council___research_tool.tavily_agent <query>")
        sys.exit(1)

    query = sys.argv[1]
    agent = TavilyAgent()

    try:
        print(f"🔍 Searching for: {query}")
        results = agent.search(query, max_results=3)

        if results.get('error'):
            print(f"❌ Error: {results['error']}")
        else:
            print(f"✅ Found {results['total_results']} results")
            if results.get('answer'):
                print(f"🤖 AI Answer: {results['answer'][:200]}...")

            for i, result in enumerate(results['results'][:3], 1):
                print(f"\n{i}. {result['title']}")
                print(f"   URL: {result['url']}")
                print(f"   Content: {result['content'][:100]}...")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
