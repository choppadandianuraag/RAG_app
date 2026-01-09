import os
import re
from typing import List, Dict, Optional, AsyncGenerator
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from langchain_core.prompts import PromptTemplate
import asyncio
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain_classic.retrievers import EnsembleRetriever
    except ImportError:
        # Fallback: use simple ensemble
        class EnsembleRetriever:
            def __init__(self, retrievers, weights):
                self.retrievers = retrievers
                self.weights = weights
            
            def invoke(self, query):
                all_docs = []
                for retriever in self.retrievers:
                    all_docs.extend(retriever.invoke(query))
                return all_docs[:10]  # Return top 10

class CrossEncoderReranker:
    """Cross-encoder reranker for improving retrieval quality."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize the cross-encoder model."""
        print(f"Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name)
        print("✓ Model loaded successfully!")
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: User's search query
            documents: List of documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            Top_k reranked documents with scores
        """
        if not documents:
            return []
        
        # Create query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Score all pairs
        scores = self.model.predict(pairs)
        
        # Sort by scores (descending)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Get top_k documents
        reranked_docs = [doc for doc, score in scored_docs[:top_k]]
        
        # Add scores to metadata
        for i, (doc, score) in enumerate(scored_docs[:top_k]):
            doc.metadata['rerank_score'] = float(score)
            doc.metadata['rerank_position'] = i + 1
        
        return reranked_docs
class RAGEngine:
    def __init__(
        self,
        # pdf_path: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        llm_model: str = "llama-3.1-8b-instant"
    ):
        # self.pdf_path = pdf_path or '/Users/anuraag/Python/RAG App/FAQ Data.pdf'
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY") or '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'
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        self.embeddings = None
        self.vector_store = None
        self.reranker = None
        self.chain = None
        self.documents = None
        self.reranking_retriever = None  

    async def initialize(self):
        """Initialize all RAG components"""
        print("  🔧 Initializing RAG Engine components...")
        
        # # 1. Load documents
        # print(f"  📄 Loading documents from: {self.pdf_path}")
        # await self._load_documents()
        
        # 2. Initialize embeddings
        print(f"  🎯 Loading embeddings: {self.embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 3. Create vector store
        print("  💾 Creating vector store...")
        await self._create_vector_store()
        
        # 4. Initialize reranker
        print("  🔀 Initializing reranker...")
        self.reranker = CrossEncoderReranker()
        
        # 5. Initialize LLM
        print(f"  🤖 Initializing LLM: {self.llm_model}")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found. Set it in environment or pass to constructor.")
        
        self.llm = ChatGroq(
            model=self.llm_model,
            groq_api_key=self.groq_api_key,
            temperature=0.1,
            max_tokens=1000
        )
        
        # 6. Build RAG chain
        print("  ⛓️ Building RAG chain...")
        await self._build_chain()
        
        print("  ✅ RAG Engine initialized successfully!")

    # async def _load_documents(self):
    #     """Load and split PDF documents"""
    #     # Check if PDF exists
    #     if not os.path.exists(self.pdf_path):
    #         raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        
    #     # Load PDF
    #     loader = PyPDFLoader(self.pdf_path)
    #     docs = loader.load()
        
    #     # Split into chunks
    #     text_splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=500,
    #         chunk_overlap=100,
    #         length_function=len,
    #         separators=["\n\n", "\n", " ", ""]
    #     )
        
    #     self.documents = text_splitter.split_documents(docs)
    #     print(f"    ✓ Loaded {len(docs)} pages, split into {len(self.documents)} chunks")
    
    async def _create_vector_store(self):
        if self.documents:
            self.vector_store = await asyncio.to_thread(
                Chroma.from_documents,
                documents=self.documents,
                embedding=self.embeddings,
                persist_directory="chroma_db",
                collection_name="faq_docs"
            )
        else:
            self.vector_store = Chroma(
                persist_directory="chroma_db",
                embedding_function=self.embeddings,
                collection_name="faq_docs"
            )

    
    async def _build_chain(self):
        """Build the RAG chain with hybrid search + reranking"""
        
        vector_retriever = self.vector_store.as_retriever(
            search_kwargs={'k': 10}
        )

        # 👉 BM25 only if documents exist
        if self.documents:
            bm25_retriever = BM25Retriever.from_documents(self.documents)
            bm25_retriever.k = 10

            hybrid_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.5, 0.5]
            )
        else:
            hybrid_retriever = vector_retriever

        
        # Reranking retriever
        def retrieve_and_rerank(query: str) -> List[Document]:
            docs = hybrid_retriever.invoke(query)
            return self.reranker.rerank(query, docs, top_k=3)
        
        reranking_retriever = RunnableLambda(retrieve_and_rerank)
        
        # Store retriever for later access
        self.reranking_retriever = reranking_retriever
        
        # Format documents
        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)
        prompt=PromptTemplate(
        template="""You are a Customer Support assiantant for a tech e-commerce.
        you are supposed to give small and accurate answers for the questions asked by the customers
        Answer ONLY from the provided Document context.
        Do NOT use any external knowledge or assumptions.
        If the context is insufficient, just say you don't know.
        {context}
        question:{question}""",
        input_variables=['context','question']
        )

        self.chain=( 
            RunnableParallel({
                'context':reranking_retriever | RunnableLambda(format_docs),
                'question':RunnablePassthrough()

            })
            | prompt
            | self.llm
            | StrOutputParser()
        )

    async def get_response(
        self,
        query: str,
        history: Optional[List] = None,
        category: Optional[str] = None
    ) -> Dict:
        """
        Get response from RAG system
        
        Args:
            query: User's question
            history: Chat history (not used currently)
            category: Optional category filter
            
        Returns:
            Dict with answer, docs, and metadata
        """
        if not self.chain:
            raise RuntimeError("RAG engine not initialized. Call initialize() first.")
        
        # Get response
        answer = await asyncio.to_thread(self.chain.invoke, query)
        
        # Get retrieved documents for transparency
        docs = await asyncio.to_thread(self.reranking_retriever.invoke, query)
        
        # Calculate confidence (based on rerank scores)
        confidence = None
        if docs and 'rerank_score' in docs[0].metadata:
            confidence = float(docs[0].metadata['rerank_score'])
        
        # Format docs for response
        doc_list = [
            {
                "content": doc.page_content[:200] + "...",
                "score": doc.metadata.get('rerank_score', 0.0),
                "source": doc.metadata.get('source', 'unknown')
            }
            for doc in docs
        ]
        
        return {
            "answer": answer,
            "docs": doc_list,
            "confidence": confidence,
            "category": category
        }
    async def get_streaming_response(
        self,
        query: str,
        history: Optional[List] = None,
        category: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Get streaming response (token by token)
        
        Args:
            query: User's question
            history: Chat history
            category: Optional category filter
            
        Yields:
            Response chunks
        """
        # For now, simulate streaming by chunking the response
        # In production, use LangChain's streaming capabilities
        result = await self.get_response(query, history, category)
        answer = result["answer"]
        
        # Split into words and yield
        words = answer.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.05)  # Simulate typing delay
    
    def load_vector_store(self):
        """Load existing Chroma DB"""
        self.vector_store = Chroma(
            persist_directory="chroma_db",
            embedding_function=self.embeddings,
            collection_name="faq_docs"
        )
        print("✓ Chroma vector store loaded")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

async def test_rag_engine():
    """Test function for RAG engine"""
    print("🧪 Testing RAG Engine...")
    
    # Initialize
    engine = RAGEngine()
    await engine.initialize()
    
    # Test queries
    test_queries = [
        "What is the refund processing time?",
        "What is the return policy during holiday season?",
        "How do I return a damaged item?"
    ]
    
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        result = await engine.get_response(query)
        print(f"💬 Answer: {result['answer']}")
        print(f"📊 Confidence: {result['confidence']:.3f}")
        print("-" * 80)

if __name__ == "__main__":
    # Test the engine
    asyncio.run(test_rag_engine())

