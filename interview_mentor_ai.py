from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import uuid, os

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

hr_prompt = """
You are a professional HR interviewer preparing a candidate for a job interview.
Ask for their name and role, then conduct a short interview.
Provide feedback after each answer.
"""

agent = create_agent(
    model=llm,
    tools=[],
    system_prompt=hr_prompt,
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

print("HR Agent Started")

while True:
    user_input = input("You: ")
    if user_input.lower() == "end":
        break

    result = agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
    )
    print(result["messages"][-1].content)
