import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma 

DATA_PATH = "data/processed" 
DB_DIR = "db/tlu_chroma"

def load_and_split_documents():
    all_final_chunks = []
    
    # 1. XỬ LÝ FILE MARKDOWN RIÊNG (QUAN TRỌNG)
    # Markdown cần tách theo Header trước để giữ ngữ cảnh
    md_loader = DirectoryLoader(DATA_PATH, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    md_docs = md_loader.load()
    
    if md_docs:
        print(f"Processing {len(md_docs)} Markdown files...")
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
        for doc in md_docs:
            # Tách theo header trước
            md_header_splits = md_splitter.split_text(doc.page_content)
            # Gán lại metadata (tên file nguồn) cho các chunk mới
            for split in md_header_splits:
                split.metadata.update(doc.metadata)
            
            # Sau đó mới split theo character nếu chunk vẫn quá dài
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_splits = text_splitter.split_documents(md_header_splits)
            all_final_chunks.extend(final_splits)

    # 2. XỬ LÝ PDF VÀ TXT (Như cũ)
    other_loaders = [
        {"glob": "**/*.pdf", "loader_cls": PyPDFLoader, "kwargs": {}},
        {"glob": "**/*.txt", "loader_cls": TextLoader, "kwargs": {"encoding": "utf-8"}},
    ]
    
    other_docs = []
    for config in other_loaders:
        loader = DirectoryLoader(DATA_PATH, glob=config["glob"], loader_cls=config["loader_cls"], loader_kwargs=config["kwargs"])
        loaded = loader.load()
        other_docs.extend(loaded)
        print(f"Loaded {len(loaded)} files from {config['glob']}")

    if other_docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        other_chunks = text_splitter.split_documents(other_docs)
        all_final_chunks.extend(other_chunks)

    print(f"\nTotal chunks created: {len(all_final_chunks)}")
    return all_final_chunks

def build_vectorstore(chunks):
    if not chunks:
        print("No chunks. Exiting.")
        return

    # Có thể dùng nomic-embed-text hoặc bge-m3 (tốt hơn cho đa ngôn ngữ)
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    
    print("Creating Vector DB...")
    # Lưu ý: Xóa DB cũ trước khi chạy lại ingest để tránh duplicate dữ liệu
    if os.path.exists(DB_DIR):
        import shutil
        # shutil.rmtree(DB_DIR) # Uncomment dòng này nếu muốn reset DB sạch sẽ mỗi lần chạy
        # print("Old DB removed.")
        pass

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name="tlu_data"
    )
    print(f"Vector DB saved to {DB_DIR}")

if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    else:
        chunks = load_and_split_documents()
        build_vectorstore(chunks)