import os
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI

# -------------------------
# Gemini Client Setup
# -------------------------

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
    http_options=types.HttpOptions(api_version="v1beta")
)

PDF_FILE_PATH = "/home/neema/Rag/ifmis.pdf"

# -------------------------
# Create File Search Store & Upload PDF
# -------------------------

display_name = "ifmis-pdf"

# Check if store already exists
existing_stores = client.file_search_stores.list()
store = None
for s in existing_stores:
    if s.display_name == display_name:
        store = s
        print(f"Using existing store: {store.name}")
        break

# Create new store if not exists
if store is None:
    store = client.file_search_stores.create(
        config={"display_name": display_name}
    )
    print(f"Created new store: {store.name}")

# Upload file to store (if not already uploaded)
file_already_uploaded = False
try:
    existing_files = list(client.files.list())
    for ef in existing_files:
        if "ifmis.pdf" in ef.name:
            file_already_uploaded = True
            print(f"File already uploaded: {ef.name}")
            break
except:
    pass

if not file_already_uploaded:
    print("Uploading PDF to file search store...")
    operation = client.file_search_stores.upload_to_file_search_store(
        file=PDF_FILE_PATH,
        file_search_store_name=store.name
    )
    
    # Wait for upload to complete
    import time
    while not operation.done:
        time.sleep(2)
        operation = client.operations.get(operation)
    
    print("PDF uploaded successfully!")

# -------------------------
# LLM Setup (Gemini)
# -------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-001",
    temperature=0
)

# Bind file search tool to LLM
llm_with_tools = llm.bind_tools([{
    "file_search": {
        "file_search_store_names": [store.name]
    }
}])

# -------------------------
# Interactive Chat Loop
# -------------------------

print("\n" + "="*50)
print("RAG System with Google File Search (Gemini)")
print("="*50)
print(f"File: {PDF_FILE_PATH}")
print(f"File Search Store: {store.name}")
print("Ask questions about the PDF content!")
print("="*50 + "\n")

while True:
    question = input("\nAsk a question (type 'exit' to quit): ")
    
    if question.lower() in ["exit", "quit"]:
        break
    
    # Use native Google client with file search tool
    response = client.models.generate_content(
        model="gemini-1.5-flash-001",
        contents=question,
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[store.name]
                    )
                )
            ]
        )
    )
    
    print("\nAnswer:")
    print(response.text)
