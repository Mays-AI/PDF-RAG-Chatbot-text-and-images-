import os
import fitz  # PyMuPDF
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from dotenv import load_dotenv
import faiss
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)

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
all_page_numbers = []  # Tracks logical (printed) page numbers for text chunks
faiss_index = faiss.IndexFlatL2(1536)
all_images = []  # Stores extracted images
image_index = []  # Maps text chunks to images


def extract_printed_page_number_from_footer(page, pdf_path):
    """
    Extracts the printed page number from the footer of a given page using layout model analysis.
    """
    footer_text = ""
    try:
        # Perform layout analysis using Azure Form Recognizer
        print(f"Analyzing footer for page {page.number + 1} of {pdf_path}")
        poller = document_intelligence_client.begin_analyze_document(
            model_id="prebuilt-layout",
            document=open(pdf_path, "rb")
        )
        result = poller.result()

        # Loop through all elements in the footer area
        for page_result in result.pages:
            for line in page_result.lines:
                # Check if the line of text is located in the footer area of the page
                if line.polygon[3][1] > 0.8 * page.rect.height:  # Footer check (bottom 20%)
                    footer_text = line.content.strip()
                    print(f"Footer text found: {footer_text}")

                    # Check if the footer text is a number (i.e., page number)
                    if footer_text.isdigit():
                        print(f"Printed page number found: {footer_text}")
                        return int(footer_text)
                    elif footer_text.lower().replace(' ', '').isalpha():  # Roman numeral check
                        print(f"Roman numeral page number found: {footer_text}")
                        return footer_text.upper()
        print("No printed page number found in footer.")
        return None  # If no valid page number is found
    except Exception as e:
        print(f"Error extracting printed page number from footer: {e}")
        return None


def analyze_pdf_layout(pdf_path):
    """
    Extracts text, images, and printed page numbers from each page of a PDF.
    """
    extracted_text_per_page = []
    extracted_images_per_page = []
    printed_page_numbers = []  # To store extracted logical page numbers

    try:
        pdf_document = fitz.open(pdf_path)

        for page_number in range(len(pdf_document)):
            page = pdf_document[page_number]
            print(f"Processing page {page_number + 1}")

            try:
                # Extract the printed page number from the footer
                printed_page_number = extract_printed_page_number_from_footer(page, pdf_path)
                printed_page_numbers.append(printed_page_number)
                print(f"Extracted printed page number: {printed_page_number}")

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
                printed_page_numbers.append(None)

    except Exception as e:
        print(f"Error opening PDF: {e}")

    return extracted_text_per_page, extracted_images_per_page, printed_page_numbers


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


def setup_pdf_index(pdf_paths):
    """
    Processes PDFs to extract text, images, and logical page numbers.
    """
    global all_text_chunks, all_page_numbers, all_images, faiss_index, image_index
    for pdf_path in pdf_paths:
        print(f"Processing PDF: {pdf_path}")
        text_per_page, images_per_page, printed_page_numbers = analyze_pdf_layout(pdf_path)

        for abs_page_number, (page_text, printed_page_number) in enumerate(zip(text_per_page, printed_page_numbers), start=1):
            if page_text.strip():  # Only process non-empty pages
                chunks = chunk_text(page_text)
                embeddings = np.array([get_embeddings(chunk) for chunk in chunks])

                # Map text chunks to printed page numbers if available, else fallback to absolute numbers
                all_text_chunks.extend(chunks)
                logical_page_number = printed_page_number if printed_page_number else abs_page_number
                all_page_numbers.extend([logical_page_number] * len(chunks))

                # Map images to chunks
                all_images.extend(images_per_page)
                for _ in chunks:
                    image_index.append(images_per_page[abs_page_number - 1])

                # Add embeddings to FAISS index
                if embeddings.size > 0:
                    faiss_index.add(embeddings.astype(np.float32))
                    print(f"Added {len(embeddings)} embeddings to FAISS index.")


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
    """
    Handles chat queries and provides answers along with page numbers and images.
    """
    global all_text_chunks, faiss_index, all_page_numbers, image_index
    data = request.get_json()
    query = data.get('query', '')

    if not query or not all_text_chunks or faiss_index is None:
        return jsonify({'answer': 'Please upload one or more PDFs first.'})

    # Perform the similarity search
    results = similarity_search(query, faiss_index, all_text_chunks, all_page_numbers, image_index)
    context = [result["text"] for result in results]
    page_numbers = [result["page"] for result in results]
    images = [result["images"] for result in results]

    # Generate answer using context
    answer = generate_answer(query, context, page_numbers)

    # Prepare image URLs
    image_urls = [os.path.basename(img) for img_list in images for img in img_list]
    if not image_urls:
        answer += "\n\nNote: No specific images were found in the context."

    return jsonify({
        'answer': answer,
        'page_numbers': page_numbers,
        'images': image_urls  # Include image URLs in the response
    })


@app.route('/static/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('static', filename)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
