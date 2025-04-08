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
from typing_extensions import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, trim_messages
from langchain.tools import Tool, tool # Use the decorator for simplicity
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.utilities import SQLDatabase
import getpass
import os
from dotenv import load_dotenv
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain import hub
from langchain_community.utilities import SQLDatabase
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
EMBEDDING_DIMENSION = 384 
EMBEDDING_MODEL = EMBEDDING_MODEL_NAME
EMBEDDING_DIM = EMBEDDING_DIMENSION

index = os.environ.get('PINECONE_INDEX_NAME')

def init_pinecone():
    api_key = os.environ.get('PINECONE_API_KEY')
    environment = os.environ.get('PINECONE_ENVIRONMENT')
    if not api_key or not environment:
        print("Pinecone API Key or Environment not configured.")
        return None
    try:
        Pinecone.init(api_key=api_key, environment=environment)
        return Pinecone
    except Exception as e:
        print(f"Failed to initialize Pinecone: {e}")
        return None

# Load Embedding Model (cache it)
# Consider loading this once globally or using a caching mechanism
embeddings = None
def get_embedding_model():
    global embeddings
    if embeddings is None:
        model_name = os.environ['EMBEDDING_MODEL']
        print(f"Loading embedding model: {model_name}")
        try:
            embeddings = SentenceTransformer(model_name)
            print("Embedding model loaded.")
        except Exception as e:
            print(f"Error loading embedding model {model_name}: {e}")
            raise e # Re-raise if model loading fails critically
    return embeddings
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)

DATABASE_URL = os.environ["DATABASE_URL"]
db = SQLDatabase.from_uri(DATABASE_URL)

query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")

# Initialize the Tavily API wrapper
search = TavilySearchAPIWrapper(max_results=5)

load_dotenv()

if "TOGETHER_API_KEY" not in os.environ:
    os.environ["TOGETHER_API_KEY"] = getpass.getpass("Enter your Together API key: ")

llm = ChatTogether(
    # together_api_key="YOUR_API_KEY",
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
)

class MessagesState(TypedDict):
    language: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    context: List[Document]
    query: str
    result: str
    answer: str
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

class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]

prompt = hub.pull("rlm/rag-prompt")

def retrieve(state: MessagesState):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: MessagesState):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

def write_query(state: MessagesState):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}

def execute_query(state: MessagesState):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}

def generate_answer(state: MessagesState):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

def decide_sql_or_rag(state: dict) -> dict:
    decision_prompt = (
        "Classify the following question as either 'SQL' if it's structured and suited for database queries, "
        "or 'RAG' if it's more general or unstructured:\n\n"
        f"Question: {state['question']}"
    )
    decision = llm.invoke(decision_prompt).strip().upper()
    if decision not in {"SQL", "RAG"}:
        decision = "RAG"
    return {"route": decision}

def check_rag_context(state: dict) -> str:
    context = state.get("context", [])
    if not context or all(len(doc.page_content) < 100 for doc in context):
        return "to_web"
    return "to_rag_answer"

# Define a new graph
workflow = StateGraph(state_schema=MessagesState)
workflow.add_sequence(
    [write_query, execute_query, generate_answer]
)
workflow.add_sequence([retrieve, generate])
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)


# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)




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