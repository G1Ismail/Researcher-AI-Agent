import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

search_tool = SerperDevTool()

def create_research_agent():
    
    Agent_LLM = ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo")
    
    return Agent(
        role="Research Specialist",
        goal="Conduct thorough and precise research on given topics",
        backstory="You are an experienced researcher with expertise in finding ans synthesizing information",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=Agent_LLM

    )

def create_research_task(agent, topic):
    return Task(
        description=f"Research the following topic and provide a comprehensive summary: {topic}",
        agent=agent,
        expected_output="A detailed summary of the research findings, including key points and insights realted to the topic",
    )

def run_research_agent(topic):
    ragent = create_research_agent()
    rtask = create_research_task(ragent, topic)
    crew = Crew(agents=[ragent], tasks=[rtask])
    result = crew.kickoff()
    return result

if __name__ == "__main__":
    print("Hi, i am your research agent, how can i help you:")
    topic = input("Enter your research topic: ")
    result = run_research_agent(topic)
    print("This is my research result:" + "\n\n" + f"{result}")