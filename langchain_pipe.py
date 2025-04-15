"""
title: LangChain Pipe
id: langchain_pipe
author: Unknown
author_url: Unknown
description: This module defines a Pipe class that utilizes LangChain
version: 0.1.0
requirements: langchain==0.3.23, langchain-core, langchain-community, pydantic==2.9.2
"""

from typing import List, Union, Generator, Iterator
from pydantic import BaseModel, Field
import os
import time
from logging import getLogger

logger = getLogger(__name__)
logger.setLevel("INFO")

# Uncomment to use OpenAI and FAISS
# from langchain_openai import ChatOpenAI
# from langchain_community.vectorstores import FAISS


class Pipeline:
    class Valves(BaseModel):
        base_url: str = Field(default="http://localhost:11434")
        ollama_model: str = Field(default="llama3.1")
        emit_interval: float = Field(
            default=2.0, description="Interval in seconds between status emissions"
        )
        enable_status_indicator: bool = Field(
            default=True, description="Enable or disable status indicator emissions"
        )

    def __init__(self):
        self.id = "langchain_pipe"
        self.name = "LangChain Pipe"
        self.valves = self.Valves()
        # self.valves = self.Valves(
        #     **{k: os.getenv(k, v.default) for k, v in self.Valves.model_fields.items()}
        # )
        self.last_emit_time = 0

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        from langchain_core.prompts.chat import ChatPromptTemplate
        from langchain.schema import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough
        from langchain_community.llms import Ollama

        # ======================================================================================================================================
        # MODEL SETUP
        # ======================================================================================================================================
        # Setup the model for generating responses
        # ==========================================================================
        # Ollama Usage
        _model = Ollama(
            model=self.valves.ollama_model,
            base_url=self.valves.base_url
        )
        # ==========================================================================
        # OpenAI Usage
        # _model = ChatOpenAI(
        #     openai_api_key=self.valves.openai_api_key,
        #     model=self.valves.openai_model
        # )
        # ==========================================================================

        # Example usage of FAISS for retreival
        # vectorstore = FAISS.from_texts(
        #     texts, embedding=OpenAIEmbeddings(openai_api_key=self.valves.openai_api_key)
        # )

        # ======================================================================================================================================
        # PROMPTS SETUP
        # ==========================================================================
        _prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful bot"),
            ("human", "{question}")
        ])
        # ======================================================================================================================================
        # CHAIN SETUP
        # ==========================================================================
        # Basic Chain
        chain = (
                {"question": RunnablePassthrough()}
                | _prompt
                | _model
                | StrOutputParser()
        )
        # ======================================================================================================================================
        # Langchain Calling
        # ======================================================================================================================================
        # messages = body.get("messages", [])
        response = ""
        # Verify a message is available
        if user_message:
            question = user_message
            try:
                # Invoke Chain
                response = chain.invoke(question)
                # Set assitant message with chain reply
                body["messages"].append({"role": "assistant", "content": response})
            except Exception as e:
                return {"error": str(e)}
        # If no message is available alert user
        else:
            body["messages"].append({"role": "assistant", "content": "No messages found in the request body"})

        return response
