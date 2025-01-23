from typing import List, Dict, Any
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class OllamaHelper:
    def __init__(self):
        # Initialize Ollama with llama2 model
        self.llm = OllamaLLM(
            model="llama3:8b",
            callbacks=[StreamingStdOutCallbackHandler()],
            temperature=0.1,
            top_k=10,
            top_p=0.95,
            repeat_penalty=1.1,
            num_ctx=4096
        )

        # Define specialized prompt templates for different query types
        self.templates = {
            "base": """You are an AI assistant using Retrieval-Augmented Generation (RAG).
            You have access to various types of documents including text files, PDFs, and images.
            You will be provided with a question and relevant context, including the type of files being referenced.
            Use only the provided context to answer the question.
            If you don't know the answer or can't find relevant information in the context, say so clearly.
            Always strive to provide concise, helpful, and context-aware answers.
            Format your response in a clear, readable way.
            
            Context:
            {context}

            Question: {question}

            Helpful Answer: Let me help you understand this based on the available information.""",

            "file_listing": """You are helping to find and list files based on specific criteria.
            Focus on:
            - File names, types, and locations
            - Creation and modification dates
            - File metadata and categories
            - Organizing files by relevance
            Format your response in a clear, readable way.
            
            Context:
            {context}

            Question: {question}

            Helpful Answer: Here are the relevant files I found:""",

            "financial_analysis": """You are analyzing financial documents and data.
            Focus on:
            - Numerical values and calculations
            - Financial periods and dates
            - Categories of income and expenses
            - Trends and patterns in financial data
            Provide clear, accurate financial information while maintaining confidentiality.
            Format your response in a clear, readable way.
            
            Context:
            {context}

            Question: {question}

            Helpful Answer: Based on the financial data, I can tell you that:""",

            "temporal_analysis": """You are analyzing information across different time periods.
            Focus on:
            - Dates and time periods
            - Changes over time
            - Historical trends
            - Temporal relationships between events or documents
            Format your response in a clear, readable way.
            
            Context:
            {context}

            Question: {question}

            Helpful Answer: Looking at the timeline of information:""",

            "person_specific": """You are finding information related to specific individuals.
            Focus on:
            - Personal documents and records
            - Individual-specific data
            - Relationships and connections
            - Personal history and timeline
            Maintain privacy and confidentiality while providing relevant information.
            Format your response in a clear, readable way.
            
            Context:
            {context}

            Question: {question}

            Helpful Answer: Regarding the person-specific information:""",

            "image_analysis": """You are analyzing information extracted from images.
            Focus on:
            - Visual elements and features
            - Text extracted from images (OCR)
            - Image metadata
            - Visual relationships and patterns
            Describe visual content clearly and accurately.
            Format your response in a clear, readable way.
            
            Context:
            {context}

            Question: {question}

            Helpful Answer: Based on the image analysis:""",

            "communication": """You are analyzing communication records and messages.
            Focus on:
            - Emails, messages, and correspondence
            - Sender and recipient information
            - Communication dates and timeline
            - Message content and context
            Maintain privacy while providing relevant information.
            Format your response in a clear, readable way.
            
            Context:
            {context}

            Question: {question}

            Helpful Answer: Regarding the communication records:""",

            "tax": """You are analyzing tax-related documents and information.
            Focus on:
            - Tax returns and filings
            - Tax periods and dates
            - Income and deductions
            - Tax calculations and assessments
            Provide accurate tax information while maintaining confidentiality.
            Format your response in a clear, readable way.
            
            Context:
            {context}

            Question: {question}

            Helpful Answer: Based on the tax documentation:""",

            "document_search": """You are searching through various documents to find specific information.
            Focus on:
            - Document content and relevance
            - Document types and categories
            - Key information and excerpts
            - Document relationships
            Format your response in a clear, readable way.
            
            Context:
            {context}

            Question: {question}

            Helpful Answer: I found the following relevant information:"""
        }

        # Create prompt templates for each type
        self.prompts = {
            key: ChatPromptTemplate.from_messages([
                ("system", template),
                ("human", "{question}")
            ])
            for key, template in self.templates.items()
        }

    def _determine_query_type(self, question: str, context: List[Dict[str, Any]]) -> str:
        """Determine the type of query based on the question and context"""
        question_lower = question.lower()
        
        # Check for image-related queries
        if any(word in question_lower for word in ["image", "picture", "photo", "show me"]):
            return "image_analysis"
        
        # Check for financial queries
        if any(word in question_lower for word in ["income", "earnings", "money", "paid", "cost", "expense"]):
            return "financial_analysis"
        
        # Check for tax-related queries
        if "tax" in question_lower:
            return "tax"
        
        # Check for time-based queries
        if any(word in question_lower for word in ["when", "year", "date", "time", "period", "over the years"]):
            return "temporal_analysis"
        
        # Check for person-specific queries
        if any(word in question_lower for word in ["my", "name", "person", "client", "daughter", "son"]):
            return "person_specific"
        
        # Check for communication queries
        if any(word in question_lower for word in ["email", "mail", "message", "sent", "received"]):
            return "communication"
        
        # Check for file listing queries
        if any(word in question_lower for word in ["list", "show", "find", "files", "documents"]):
            return "file_listing"
        
        # Check for document search queries
        if any(word in question_lower for word in ["search", "look for", "find", "where is"]):
            return "document_search"
        
        # Default to base template
        return "base"

    def create_rag_chain(self, retriever):
        """Create a RAG chain with the given retriever using LCEL"""
        # Create the document chain
        document_chain = create_stuff_documents_chain(
            self.llm,
            self.prompts["base"]  # Use base prompt as default
        )
        
        # Create the retrieval chain
        retrieval_chain = create_retrieval_chain(
            retriever,
            document_chain
        )
        return retrieval_chain

    async def get_rag_response(self, question: str, context: List[Dict[str, Any]]) -> str:
        """Get response using RAG with provided context"""
        try:
            # Format context from Neo4j results
            formatted_context = "\n\n".join([
                f"Document: {doc['text']}\n" +
                f"File Type: {doc['metadata'].get('file_type', 'unknown')}\n" +
                f"Title: {doc['metadata'].get('title', 'unknown')}\n" +
                (f"Related Documents: {', '.join([r['text'][:200] for r in doc['metadata'].get('related_documents', [])])}"
                 if doc['metadata'].get('related_documents') else "No related documents")
                for doc in context
            ])

            # Determine query type and get appropriate prompt
            query_type = self._determine_query_type(question, context)
            prompt_template = self.prompts[query_type]

            # Create a prompt with the formatted context
            prompt = prompt_template.format_messages(
                context=formatted_context,
                question=question
            )
            
            # Get response from LLM and ensure it's a string
            response = self.llm.invoke(prompt)
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, str):
                return response
            else:
                return str(response)
        except Exception as e:
            print(f"Error getting RAG response: {str(e)}")
            return "I apologize, but I encountered an error while processing your question."

    def get_direct_response(self, question: str) -> str:
        """Get a direct response without RAG context"""
        try:
            # Determine query type
            query_type = self._determine_query_type(question, [])
            prompt = self.prompts[query_type].format_messages(
                context="No direct context available.",
                question=question
            )
            response = self.llm.invoke(prompt)
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, str):
                return response
            else:
                return str(response)
        except Exception as e:
            print(f"Error getting direct response: {str(e)}")
            return "I apologize, but I encountered an error while processing your request."

    def add_custom_template(self, template_type: str, template: str):
        """Add a new custom template type"""
        try:
            self.templates[template_type] = template
            self.prompts[template_type] = ChatPromptTemplate.from_messages([
                ("system", template),
                ("human", "{question}")
            ])
        except Exception as e:
            print(f"Error adding custom template: {str(e)}")

    def update_template(self, template_type: str, template: str):
        """Update an existing template"""
        try:
            if template_type in self.templates:
                self.templates[template_type] = template
                self.prompts[template_type] = ChatPromptTemplate.from_messages([
                    ("system", template),
                    ("human", "{question}")
                ])
            else:
                print(f"Template type '{template_type}' not found")
        except Exception as e:
            print(f"Error updating template: {str(e)}")