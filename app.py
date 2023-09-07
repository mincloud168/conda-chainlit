from langchain import HuggingFaceHub, OpenAI
from langchain import PromptTemplate, LLMChain

from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents.structured_chat.prompt import SUFFIX

from qa_tools import faq_tool
from image_tools import  generate_image_tool, edit_image_tool, generate_story_tool

#from lab import query_pinecone,construtPrompt
import os

from dotenv import load_dotenv
import chainlit as cl
import requests

# Load environment variables from .env file
load_dotenv()

#HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

prompt_template = """
You are a helpful AI assistant and provide the answer for the question asked politely.

{question}
"""

#index_name = "mtnet-faq-index"

my_index = 0
my_story_part = ["一名建築工人在高處作業時，不慎失足墜落，造成嚴重頭部受傷。",
                 "工人被立即送往醫院接受治療，經過檢查發現有顱內出血和腦震盪的情況。",
                 "工地負責人被要求調查事故原因，發現缺乏安全網和適當的警示標示是主要原因。",
                 "工人康復後提起訴訟，法院判定工地負責人應負全部責任，並賠償工人的醫療費用和損失。"]

img_url1 = 'https://cdn.stability.ai/assets/org-rp8fWa6VaxuEKJ2Gw27Qm9Hh/00000000-0000-0000-0000-000000000000/bcab9956-2b6d-ecfd-7eb4-17647ff4649e'
img_url2 = 'https://cdn.stability.ai/assets/org-rp8fWa6VaxuEKJ2Gw27Qm9Hh/00000000-0000-0000-0000-000000000000/1e1db71d-0ede-af6f-01e8-ec160aa0729c'
img_url3 = 'https://cdn.stability.ai/assets/org-rp8fWa6VaxuEKJ2Gw27Qm9Hh/00000000-0000-0000-0000-000000000000/81787a51-7d82-2a35-b113-0c61cd064df1'
img_url4 = 'https://cdn.stability.ai/assets/org-rp8fWa6VaxuEKJ2Gw27Qm9Hh/00000000-0000-0000-0000-000000000000/06060668-8dca-dfc8-549f-4c8348b4a621'

my_story_images = [img_url1,img_url2,img_url3,img_url4]
my_image_name = "my_image_name"

@cl.on_chat_start
def start():
    llm = ChatOpenAI(temperature=0, streaming=True)

    #tools = [faq_tool]    
    tools = [faq_tool]#[faq_tool, generate_image_tool, edit_image_tool]
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
    load_images()



def load_images():
    i = 0
    # Fetch the image data from the blob URL
    for img in my_story_images:
        response = requests.get(img)
        if response.status_code == 200:
            img_name = my_image_name + str(i)
            cl.user_session.set(img_name,response.content)
            print(img_name)
        i+=1
        

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

    #if my_img:
    #    print(my_img)
    #else:
    #    print("my_img null")

    if generated_image:       
        elements = [
            cl.Image(
                content=generated_image,
                name=generated_image_name,
                display="inline",
            )           
        ]

    elements = [cl.Pdf(name="Doc1", display="inline", path="./labor_case_001.pdf"),
                cl.Pdf(name="Doc2", display="inline", path="./labor_case_000.pdf")]
    #actions = [cl.Action(name="next_story", value="0")]

    await cl.Message(content=res, elements=elements, actions=actions).send()
    

@cl.action_callback("next_story")
async def on_action(action):
    #current_index = my_index
    generated_image = cl.user_session.get(my_image_name + action.value)
    elements = []
    actions = []
    if generated_image:
        elements = [
            cl.Image(
                content=generated_image,
                name=my_image_name,
                display="inline",
            )           
        ]

    my_index = int(action.value)
    next_story = my_story_part[my_index]
    #prepare next
    
    if my_index == len(my_story_part) - 1:
        actions = [cl.Action(name="publish", value="https://drive.google.com/file/d/1UoTA4OrK8LAu19iiPMxbAsrXvjEKIQVJ/view?usp=sharing")]
    else:
        my_index+=1
        actions = [cl.Action(name="next_story", value=str(my_index))]
        
    
    await cl.Message(content=f"{next_story}",elements=elements,actions=actions).send()

@cl.action_callback("publish")
async def on_action(action):
    await cl.Message(content=f"{action.value}").send()

"""
@cl.langchain_run
async def run(agent, input_str):
    res = await agent.acall(input_str, callbacks=[cl.AsyncChainlitCallbackHandler()])
    await cl.Message(content=res).send()
"""

"""
    repo_id = "tiiuae/falcon-7b-instruct"
    llm = HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN, 
                        repo_id=repo_id, 
                        model_kwargs={"temperature":0.7, "max_new_tokens":500})
"""


