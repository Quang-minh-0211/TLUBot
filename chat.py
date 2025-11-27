# chat.py
# Sửa lại các import
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
# Thêm các module cần thiết cho LCEL
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser # Để parse output LLM thành chuỗi

DB_DIR = "db/tlu_chroma"

# Hàm chuyển danh sách documents thành chuỗi text cho Prompt
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain():
    # 1) Load vector DB + retriever
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )

    vectordb = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name="tlu_data"
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 10})

    # 2) LLM
    llm = ChatOllama(
        model="qwen2.5:7b",
        base_url="http://localhost:11434",
        temperature=0.2,
    )

    # 3) Prompt
    system_prompt = """
Bạn là chatbot tư vấn cho học sinh cấp 3 về Trường Đại học Thủy Lợi.
Chỉ sử dụng thông tin trong phần CONTEXT để trả lời.
Nếu trong context không có thông tin, hãy nói lịch sự rằng bạn không có đủ dữ liệu để trả lời chính xác
và gợi ý học sinh truy cập website chính thức của trường.

CONTEXT:
{context}
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # 4) XÂY DỰNG RAG CHAIN BẰNG LCEL (THAY THẾ create_stuff_documents_chain VÀ create_retrieval_chain)
    
    # Gán input của người dùng (input) và kết quả tìm kiếm (context) vào prompt
    rag_chain = (
        # Bước 1: Lấy input của người dùng và gán vào 2 khóa: input và context
        {
            "context": retriever | RunnableLambda(format_docs), # Lấy docs từ retriever và format thành chuỗi text
            "input": RunnablePassthrough() # Lấy nguyên input (câu hỏi)
        }
        | prompt # Bước 2: Truyền vào Prompt Template
        | llm    # Bước 3: Truyền vào LLM
        | StrOutputParser() # Bước 4: Lấy kết quả dưới dạng chuỗi
    )

    return rag_chain


def main():
    chain = get_rag_chain()
    print("Chatbot ĐH Thủy Lợi (gõ 'exit' để thoát)\n")

    while True:
        question = input("Bạn: ")
        if question.lower().strip() in ["exit", "quit"]:
            break

        # LCEL chain trả về trực tiếp string, không phải dictionary
        result = chain.invoke(question) 
        
        # Vì result là string, ta in trực tiếp
        print("\nBot:", result, "\n")


if __name__ == "__main__":
    # Bạn cần cài thêm 1 thư viện nữa nếu chưa có:
    # pip install langchain-core
    main()