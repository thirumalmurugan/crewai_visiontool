import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_visiontool.tools.yolo_tool import YoloTool
from dotenv import load_dotenv

load_dotenv()

@CrewBase
class CrewaiVisiontool():
    """CrewaiVisiontool crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def object_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['object_analyst'],
            tools=[YoloTool()],
            verbose=True,
            max_iter=3,
            llm=LLM(
                model=os.getenv("MODEL", "ollama/llama3.2"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            )
        )

    @task
    def detect_objects_task(self) -> Task:
        return Task(
            config=self.tasks_config['detect_objects_task'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the CrewaiVisiontool crew"""
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )

