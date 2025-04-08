from typing import Annotated
from typing import Sequence
import io
import sys
from langchain_together import ChatTogether
import json
from pathlib import Path
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, trim_messages
from langchain.tools import Tool, tool # Use the decorator for simplicity
from langchain_core.pydantic_v1 import BaseModel, Field
import getpass
import os
from dotenv import load_dotenv
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

# Initialize the Tavily API wrapper
search = TavilySearchAPIWrapper(max_results=5)

load_dotenv()

if "TOGETHER_API_KEY" not in os.environ:
    os.environ["TOGETHER_API_KEY"] = getpass.getpass("Enter your Together API key: ")

llm = ChatTogether(
    # together_api_key="YOUR_API_KEY",
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
)

messages_store = {}
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are trying to lower student stress level to the best of your ability. Student is feeling {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
# Define a new graph
workflow = StateGraph(state_schema=MessagesState)


# Define the function that calls the model
def call_model(state: MessagesState):
    prompt = prompt_template.invoke(
        {"messages": state["messages"], "language": state.get("language", "en")} # , "language": state["language"]
    )
    response = llm.invoke(prompt)
    return {"messages": response}

def tavily_web_search(query: str) -> list[str]:
    results = search.results(query)
    return [result['content'] for result in results if 'content' in result]

class PineconeQueryInput(BaseModel):
    query: str = Field(description="The user's question or topic to search for in the text documents.")

class SQLQueryInput(BaseModel):
    sql_query: str = Field(description="A valid and safe PostgreSQL SELECT query to execute against the relevant table(s). Query ONLY the 'uploaded_tabular_data' table, filtering by 'upload_id' and 'row_index' or querying the 'row_data' JSONB column.")

class ProposeWriteInput(BaseModel):
    target_upload_id: int = Field(description="The specific ID of the tabular data upload to modify.")
    write_request: str = Field(description="A detailed natural language description of the data to be inserted, updated, or deleted. Include specific values and conditions.")

# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

class MessagesState(TypedDict):
    language: str
    messages: Annotated[Sequence[BaseMessage], add_messages]


def conversational_rag_chain(input, id):
    config = {"configurable": {"thread_id": id}}
    query = input["input"]
    lang = input["context"]
    input_messages = [HumanMessage(query)]

    output = app.invoke({
        "messages": input_messages, "language": lang},
        config)

    # Capture the pretty-printed output
    buffer = io.StringIO()
    sys.stdout = buffer  # Redirect stdout to the buffer
    output["messages"][-1].pretty_print()
    sys.stdout = sys.__stdout__  # Reset stdout to normal

    pretty_output = buffer.getvalue()  # Get the captured output as a string
    return pretty_output


if __name__ == "__main__":
    while True :
        query = input("Input: ")
        lang = "Angry"
        input_messages = [HumanMessage(query)]

        # Define the missing config
        config = {"configurable": {"thread_id": 1234}}  # Replace "default_thread" with an actual ID

        output = app.invoke(
            {"messages": input_messages, "language": lang},
            config
        )
        output["messages"][-1].pretty_print()  # output contains all messages in state