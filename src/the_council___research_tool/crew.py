import os

from crewai import LLM
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import (
	ScrapeWebsiteTool,
	ArxivPaperTool,
	FileReadTool
)





@CrewBase
class TheCouncilResearchToolCrew:
    """TheCouncilResearchTool crew"""

    
    @agent
    def research_content_processor(self) -> Agent:
        
        return Agent(
            config=self.agents_config["research_content_processor"],
            
            
            tools=[				FileReadTool()],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            max_execution_time=None,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
            
        )
    
    @agent
    def research_summarizer(self) -> Agent:
        
        return Agent(
            config=self.agents_config["research_summarizer"],
            
            
            tools=[],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            max_execution_time=None,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
            
        )
    
    @agent
    def external_research_specialist(self) -> Agent:
        
        return Agent(
            config=self.agents_config["external_research_specialist"],
            
            
            tools=[				ScrapeWebsiteTool(),
				ArxivPaperTool()],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            max_execution_time=None,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
            
        )
    
    @agent
    def research_critic_and_evaluator(self) -> Agent:
        
        return Agent(
            config=self.agents_config["research_critic_and_evaluator"],
            
            
            tools=[],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            max_execution_time=None,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
            
        )
    
    @agent
    def research_gap_analyst(self) -> Agent:
        
        return Agent(
            config=self.agents_config["research_gap_analyst"],
            
            
            tools=[],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            max_execution_time=None,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
            
        )
    
    @agent
    def research_synthesizer(self) -> Agent:
        
        return Agent(
            config=self.agents_config["research_synthesizer"],
            
            
            tools=[],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            max_execution_time=None,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
            
        )
    

    
    @task
    def process_research_content(self) -> Task:
        return Task(
            config=self.tasks_config["process_research_content"],
            markdown=False,
            
            
        )
    
    @task
    def summarize_research_content(self) -> Task:
        return Task(
            config=self.tasks_config["summarize_research_content"],
            markdown=False,
            
            
        )
    
    @task
    def conduct_external_research(self) -> Task:
        return Task(
            config=self.tasks_config["conduct_external_research"],
            markdown=False,
            
            
        )
    
    @task
    def critique_research_quality(self) -> Task:
        return Task(
            config=self.tasks_config["critique_research_quality"],
            markdown=False,
            
            
        )
    
    @task
    def identify_research_gaps(self) -> Task:
        return Task(
            config=self.tasks_config["identify_research_gaps"],
            markdown=False,
            
            
        )
    
    @task
    def synthesize_research_findings(self) -> Task:
        return Task(
            config=self.tasks_config["synthesize_research_findings"],
            markdown=False,
            
            
        )
    

    @crew
    def crew(self) -> Crew:
        """Creates the TheCouncilResearchTool crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )

    def _load_response_format(self, name):
        with open(os.path.join(self.base_directory, "config", f"{name}.json")) as f:
            json_schema = json.loads(f.read())

        return SchemaConverter.build(json_schema)
