"""
Test script to verify RAG engine works with real PDFs
"""

from rag_engine import RAGEngine

def test_rag():
    print("=" * 50)
    print("Testing RAG Engines with real PDF")
    print("=" * 50)

    # Initializing Engine

    engine = RAGEngine()

    # Step 1 Load the document

    print("\n Step 1 : Loading PDF...")
    file_path = "constitution_of_india.pdf"

    documents = engine.load_documents([file_path])

    if not documents:
        print("Failed to load documents!")
        return
    
    print(f"Successfully loaded {len(documents)} chunks")

    
    # Step 2 : Create Vector database

    print("\n Step 2 : Creating embeddings")
    print("This will take 30-60 seconds")

    engine.create_vector_store(documents)

    print("Vector store created")


    # Step 3 : Test Queries
    
    print("\n" + "=" * 50)
    print("Testing Queries")
    print("=" * 50)

    test_questions = [ 
        "What is this document about",
        "What is citizenship",
        "what are directive principal of state policy",
    ]


    for i, question in enumerate(test_questions, 1):
        print(f"\n Question {i}: {question}")
        print("-" * 50)

        response = engine.query(question, k=3)

        print(f"Answer: \n{response["answer"]}\n")
        print(f"Sources used : {len(response['sources'])} chunks ")

        # Show First source
        if response['sources']:
            first_source = response['sources'][0]
            print(f"From : {first_source['source']} (Page {first_source['page']} ")

    # Step 4 : Test Persistence
    print("\n" + "=" * 50)
    print("Testing Persistence")
    print("=" * 50)

    print("\n Creating New engine instance....")

    engine2 = RAGEngine()

    if engine2.load_existing_vectorstore():
        print("Successfully Loaded from disk")

        # Quick test

        response = engine2.query(" What are the fundamentak rights ?", k=2)
        print(f"\n Quick test answer : {response["answer"][:200]}")
    else:
        print("Failed to load from the disk")


    print("\n" + "=" * 50)
    print("All tests complete!")
    print("=" * 50)   

if __name__ == "__main__":
    test_rag() 


