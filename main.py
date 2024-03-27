from crewai import Crew, Process

from agents import AINewsletterAgents
from tasks import AINewsletterTasks
from langchain_openai import ChatOpenAI

from file_io import save_markdown

from dotenv import load_dotenv
load_dotenv()

OpenAIGPT4 = ChatOpenAI(
	model="gpt-4"
	)

agents = AINewsletterAgents()
tasks = AINewsletterTasks()

#Setting up agents
editor = agents.editor_agent()
news_fetcher = agents.news_fetcher_agent()
news_analyzer = agents.news_analyzer_agent()
newsletter_compiler = agents.newsletter_compiler_agent()

#Setting up tasks
fetch_news_task = tasks.fetch_news_task(news_fetcher)
analyzed_news_task = tasks.analyze_news_task(news_analyzer, [fetch_news_task])
compiled_news_task = tasks.compile_newsletter_task(newsletter_compiler, [analyzed_news_task], callback_function=save_markdown)

#Setting up tools - done in tools

crew = Crew(
	agents = [editor, news_fetcher, news_analyzer, newsletter_compiler],
	tasks = [fetch_news_task, analyzed_news_task, compiled_news_task],
	process = Process.hierarchical,
	manager_llm = OpenAIGPT4
	)

# Kick off the crew
results = crew.kickoff()

print("Crew work results:")
print(results)