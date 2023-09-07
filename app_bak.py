from langchain import HuggingFaceHub, OpenAI
from langchain import PromptTemplate, LLMChain

from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents.structured_chat.prompt import SUFFIX

from qa_tools import faq_tool
from image_tools import generate_image_tool, edit_image_tool

#from lab import query_pinecone,construtPrompt
import os

from dotenv import load_dotenv
import chainlit as cl

# Load environment variables from .env file
load_dotenv()

#HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

prompt_template = """
You are a helpful AI assistant and provide the answer for the question asked politely.

{question}
"""

#index_name = "mtnet-faq-index"

@cl.on_chat_start
def start():
    llm = ChatOpenAI(temperature=0, streaming=True)

    #tools = [faq_tool]    
    tools = [faq_tool, generate_image_tool, edit_image_tool]
    memory = ConversationBufferMemory(memory_key="chat_history")
    _SUFFIX = "Chat history:\n{chat_history}\n\n" + SUFFIX

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        agent_kwargs={
            "suffix": _SUFFIX,
            "input_variables": ["input", "agent_scratchpad", "chat_history"],
        },
    )
    cl.user_session.set("agent", agent)



@cl.on_message
async def main(message):
    #
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    cl.user_session.set("generated_image", None)

    res = await cl.make_async(agent.run)(
        input=message, callbacks=[cl.LangchainCallbackHandler()]
    )
    elements = []
    actions = []
    generated_image_name = cl.user_session.get("generated_image")
    generated_image = cl.user_session.get(generated_image_name)
    if generated_image:
        elements = [
            cl.Image(
                content=generated_image,
                name=generated_image_name,
                display="inline",
            )
        ]
        actions = [cl.Action(name="Create variation", value=generated_image_name)]

    await cl.Message(content=res, elements=elements, actions=actions).send()