from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from morph_lib.stream import stream_chat

import morph
from morph import MorphGlobalContext
import os

history = {}

@morph.func
def langchain_chat(context: MorphGlobalContext):
    groq_api_key = os.environ["GROQ_API_KEY"]
    llm = ChatGroq(model="deepseek-r1-distill-qwen-32b", api_key=groq_api_key)

    if history.get("thread_id") != context.vars["thread_id"]:
        history["thread_id"] = context.vars["thread_id"]
        history["messages"] = []
    history["messages"].append(HumanMessage(content=context.vars["prompt"]))
    result = ""
    for token in llm.stream(history["messages"]):
        yield stream_chat(token.content)
    history["messages"].append(AIMessage(content=result))
