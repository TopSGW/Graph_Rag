from typing import List, Dict, Any
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class OllamaHelper:
    def __init__(self):
        # Initialize Ollama with llama3.3 70b model
        self.llm = Ollama(
            model="llama3.3:70b",
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            temperature=0.1,
            top_k=10,
            top_p=0.95,
            repeat_penalty=1.1
        )

        # Define RAG prompt template
        self.rag_template = """You are an AI assistant using Retrieval-Augmented Generation (RAG).
        RAG enhances your responses by retrieving relevant information from a knowledge base.
        You will be provided with a question and relevant context. Use only this context to answer the question.
        Do not make up an answer. If you don't know the answer, say so clearly.
        Always strive to provide concise, helpful, and context-aware answers.
        
        Context:
        {context}

        Question: {question}

        Helpful Answer: Let me help you understand this accounting concept."""

        self.qa_template = """You are an AI assistant using Retrieval-Augmented Generation (RAG). Answer the following question based on your knowledge.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        
        Question: {question}

        Helpful Answer: Let me help you understand this accounting concept."""

        self.rag_prompt = PromptTemplate(
            template=self.rag_template,
            input_variables=["context", "question"]
        )

        self.qa_prompt = PromptTemplate(
            template=self.qa_template,
            input_variables=["question"]
        )

    def create_rag_chain(self, retriever):
        """Create a RAG chain with the given retriever"""
        # Define the RAG chain
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | self.rag_prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

    async def get_rag_response(self, question: str, context: List[Dict[str, Any]]) -> str:
        """Get response using RAG with provided context"""
        try:
            # Format context from Neo4j results
            formatted_context = "\n\n".join([
                f"Document: {doc['text']}\n" +
                (f"Related Documents: {', '.join([r['text'][:200] for r in doc['metadata'].get('related_documents', [])])}"
                 if doc['metadata'].get('related_documents') else "No related documents")
                for doc in context
            ])

            # Format prompt with context
            prompt = self.rag_prompt.format(
                context=formatted_context,
                question=question
            )
            
            # Get response from LLM
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            print(f"Error getting RAG response: {str(e)}")
            return "I apologize, but I encountered an error while processing your question."

    def get_direct_response(self, question: str) -> str:
        """Get a direct response without RAG context"""
        try:
            prompt = self.qa_prompt.format(question=question)
            return self.llm.invoke(prompt)
        except Exception as e:
            print(f"Error getting direct response: {str(e)}")
            return "I apologize, but I encountered an error while processing your request."

    def create_retrieval_qa(self, retriever) -> RetrievalQA:
        """Create a RetrievalQA chain"""
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": self.rag_prompt
            }
        )

    def set_custom_rag_template(self, template: str):
        """Set a custom RAG template"""
        try:
            self.rag_template = template
            self.rag_prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
        except Exception as e:
            print(f"Error setting custom RAG template: {str(e)}")
            # Revert to default template
            self.rag_prompt = PromptTemplate(
                template=self.rag_template,
                input_variables=["context", "question"]
            )