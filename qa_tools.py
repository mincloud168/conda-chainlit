from langchain.tools import Tool, StructuredTool

from lab import query_pinecone,construtPrompt

import openai

COMPLETIONS_MODEL = "text-davinci-003"

index_name = "mtnet-faq-index"

def question_answer(message:str):
    contexts = query_pinecone(query=message,index_name=index_name,text_key="content")
    prompt_contexts = construtPrompt(query=message,contexts=contexts)
    answer = complete(prompt=prompt_contexts)

    return answer

faq_tool = StructuredTool.from_function(
    func=question_answer,
    name="FAQTool",
    description="MTNet questions and answers people often ask. MTNet相關的問題與答案",
    return_direct=True,
)

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 1000,
    "model": COMPLETIONS_MODEL,
}


# let's make it simpler to get answers
def complete(prompt):
    # query text-davinci-003
    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )
    print(prompt)
    return response["choices"][0]["text"].strip(" \n")