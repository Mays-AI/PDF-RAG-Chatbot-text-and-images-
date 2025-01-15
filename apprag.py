import os
import fitz  # PyMuPDF
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from dotenv import load_dotenv
import faiss
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__, static_folder="static")

load_dotenv()

# Azure OpenAI and Form Recognizer clients
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OAI_KEY"),
    api_version="2024-06-01",
    azure_endpoint=os.getenv("AZURE_OAI_ENDPOINT")
)

doc_intelligence_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
doc_intelligence_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
document_intelligence_client = DocumentAnalysisClient(
    endpoint=doc_intelligence_endpoint,
    credential=AzureKeyCredential(doc_intelligence_key)
)

# Global variables
all_text_chunks = []
all_page_numbers = []  # Tracks PDF page numbers for text chunks
faiss_index = faiss.IndexFlatL2(1536)
all_images = []  # Stores extracted images
image_index = []  # Maps text chunks to images


# Function to analyze PDF layout and extract text and images
def analyze_pdf_layout(pdf_path):
    """
    Extracts text and images from each page of a PDF.
    Uses the PDF page number only, without attempting to extract printed page numbers.
    """
    extracted_text_per_page = []
    extracted_images_per_page = []

    try:
        pdf_document = fitz.open(pdf_path)

        for page_number in range(len(pdf_document)):
            page = pdf_document[page_number]
            print(f"Processing page {page_number + 1}")

            try:
                # Analyze layout to extract text using Azure Form Recognizer layout model
                poller = document_intelligence_client.begin_analyze_document(
                    model_id="prebuilt-layout",
                    document=open(pdf_path, "rb")
                )
                result = poller.result()

                # Extract text for the current page
                page_text = "\n".join([line.content for line in result.pages[page_number].lines])
                extracted_text_per_page.append(page_text)

                # Extract images for the current page
                images = page.get_images(full=True)
                page_images = []
                for img_index, image in enumerate(images):
                    xref = image[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image_path = f"static/images/page_{page_number + 1}_img_{img_index}.{image_ext}"
                    os.makedirs("static/images", exist_ok=True)
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    page_images.append(image_path)
                extracted_images_per_page.append(page_images)

                print(f"Processed page {page_number + 1}")

            except Exception as e:
                print(f"Error processing page {page_number + 1}: {e}")
                extracted_text_per_page.append("")
                extracted_images_per_page.append([])

    except Exception as e:
        print(f"Error opening PDF: {e}")

    return extracted_text_per_page, extracted_images_per_page


# Function to chunk text for better embedding
def chunk_text(text, max_chunk_size=500):
    """
    Splits text into smaller chunks for processing.
    """
    chunks = []
    words = text.split()
    current_chunk = []
    for word in words:
        if len(" ".join(current_chunk + [word])) > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
        else:
            current_chunk.append(word)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    print(f"Chunked text into {len(chunks)} chunks.")
    return chunks


# Function to generate embeddings for text
def get_embeddings(text):
    """
    Generates embeddings for a given text chunk.
    """
    print(f"Generating embeddings for text: {text[:50]}...")
    response = client.embeddings.create(
        input=text,
        model="embed_chat"
    )
    embeddings = np.array(response.data[0].embedding)
    return embeddings


# Function to process PDFs and create FAISS index
def setup_pdf_index(pdf_paths):
    """
    Processes PDFs to extract text, images, and PDF page numbers.
    """
    global all_text_chunks, all_page_numbers, all_images, faiss_index, image_index
    for pdf_path in pdf_paths:
        print(f"Processing PDF: {pdf_path}")
        text_per_page, images_per_page = analyze_pdf_layout(pdf_path)

        for abs_page_number, page_text in enumerate(text_per_page, start=1):
            if page_text.strip():  # Only process non-empty pages
                chunks = chunk_text(page_text)
                embeddings = np.array([get_embeddings(chunk) for chunk in chunks])

                # Map text chunks to PDF page numbers
                all_text_chunks.extend(chunks)
                all_page_numbers.extend([abs_page_number] * len(chunks))

                # Map images to chunks
                all_images.extend(images_per_page)
                for _ in chunks:
                    image_index.append(images_per_page[abs_page_number - 1])

                # Add embeddings to FAISS index
                if embeddings.size > 0:
                    faiss_index.add(embeddings.astype(np.float32))
                    print(f"Added {len(embeddings)} embeddings to FAISS index.")


# Function to perform similarity search
def similarity_search(query, faiss_index, text_chunks, page_numbers, image_index):
    """
    Searches for the most relevant text chunks and retrieves associated data.
    """
    print(f"Performing similarity search for query: {query}")
    query_embedding = get_embeddings(query).reshape(1, -1).astype(np.float32)
    distances, indices = faiss_index.search(query_embedding, k=2)
    results = []

    for i in indices[0]:
        if 0 <= i < len(text_chunks):
            result = {
                "text": text_chunks[i],
                "page": page_numbers[i] if i < len(page_numbers) else None,
                "images": image_index[i] if i < len(image_index) else []
            }
            print(f"Match found: {result}")
            results.append(result)
        else:
            print(f"Warning: Index {i} is out of bounds. Skipping this result.")

    return results


# Function to generate an answer using context
def generate_answer(query, context, page_numbers):
    """
    Generates a response using the query and associated context.
    """
    context_with_pages = "\n".join(
        [f"Page {page}: {text}" for text, page in zip(context, page_numbers)]
    )
    print(f"Generating answer with context:\n{context_with_pages[:500]}")
    response = client.chat.completions.create(
        model="gpt-4o-2",
        messages=[
            {"role": "user", "content": f"Using the following context, answer the question: '{query}'\n\nContext: {context_with_pages}"}
        ],
        temperature=0.7,
        max_tokens=150
    )
    answer = response.choices[0].message.content
    page_number_list = ", ".join(map(str, page_numbers))
    final_answer = f"{answer}\n\n(Answer derived from pages: {page_number_list})"
    print(f"Generated answer: {final_answer}")
    return final_answer


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_pdfs():
    """
    Handles PDF uploads and processes them.
    """
    global all_text_chunks, faiss_index, all_images, image_index, all_page_numbers
    if 'files' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    files = request.files.getlist('files')
    pdf_paths = []

    for file in files:
        if file.filename == '':
            continue
        pdf_path = os.path.join("upload", file.filename)
        os.makedirs("upload", exist_ok=True)
        file.save(pdf_path)
        pdf_paths.append(pdf_path)

    setup_pdf_index(pdf_paths)

    return jsonify({'message': 'PDFs uploaded and processed successfully.', 'images': all_images})


@app.route('/chat', methods=['POST'])
def chat_with_pdfs():
    global all_text_chunks, faiss_index, all_page_numbers, image_index
    data = request.get_json()
    query = data.get('query', '')

    if not query or not all_text_chunks or faiss_index is None:
        return jsonify({'answer': 'Please upload one or more PDFs first.'})

    results = similarity_search(query, faiss_index, all_text_chunks, all_page_numbers, image_index)
    context = [result["text"] for result in results]
    page_numbers = [result["page"] for result in results]
    images = [result["images"] for result in results]

    # Flatten and deduplicate images
    all_image_paths = [img for img_list in images for img in img_list]
    unique_image_paths = list(set(all_image_paths))  # Remove duplicates

    answer = generate_answer(query, context, page_numbers)

    # Prepare image URLs
    image_urls = [f"static/images/{os.path.basename(img)}" for img in unique_image_paths]
    if not image_urls:
        answer += "\n\nNote: No specific images were found in the context."

    return jsonify({'answer': answer, 'images': image_urls})


if __name__ == '__main__':
    app.run(debug=True)
