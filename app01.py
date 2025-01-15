import os
import fitz  # PyMuPDF
from openai import AzureOpenAI
from dotenv import load_dotenv
import faiss
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OAI_KEY"),
    api_version="2024-06-01",
    azure_endpoint=os.getenv("AZURE_OAI_ENDPOINT")
)

# Initialize global variables
all_text_chunks = []  # Global to store text chunks from all PDFs
faiss_index = faiss.IndexFlatL2(1536)  # Placeholder, replace 1536 with your embedding dimension
all_images = []  # Global list for images from all PDFs

# Function to extract text and images from PDF
def extract_text_and_images_from_pdf(pdf_path):
    text = ""
    extracted_images = []  # List to store extracted images
    with fitz.open(pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text += page.get_text("text")
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = pdf.extract_image(xref)
                image_bytes = base_image["image"]
                image_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_image_{page_num}_{img_index}.png"
                extracted_images.append(image_filename)
                save_image(image_bytes, image_filename)
    return text, extracted_images

# Function to save images
def save_image(image_data, filename):
    static_path = os.path.join('static', 'extracted_images', filename)
    os.makedirs(os.path.dirname(static_path), exist_ok=True)
    with open(static_path, 'wb') as f:
        f.write(image_data)
    print(f"Saved image: {static_path}")

# Function to chunk text
def chunk_text(text, max_chunk_size=200):
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
    return chunks

# Function to get embeddings
def get_embeddings(text):
    response = client.embeddings.create(
        input=text,
        model="embed_chat"
    )
    embeddings = np.array(response.data[0].embedding)
    return embeddings

# Function to setup PDF index
def setup_pdf_index(pdf_paths):
    global all_text_chunks, all_images, faiss_index
    for pdf_path in pdf_paths:
        extracted_text, extracted_images = extract_text_and_images_from_pdf(pdf_path)
        text_chunks = chunk_text(extracted_text)

        # Get embeddings for each chunk
        embeddings = np.array([get_embeddings(chunk) for chunk in text_chunks])
        
        # Update global lists and index
        all_text_chunks.extend(text_chunks)
        all_images.extend(extracted_images)
        
        # Ensure embeddings are added to the index correctly
        if embeddings.shape[0] > 0:  # Ensure there are embeddings to add
            faiss_index.add(embeddings.astype(np.float32))  # Ensure the correct type

        # Debugging: Check the structure
        print("Debug Info:")
        print("Total number of vectors in FAISS index:", faiss_index.ntotal)
        
        # Cross-check first 5 vectors with their text and images
        for i in range(min(5, faiss_index.ntotal)):
            print(f"Text chunk {i}: {all_text_chunks[i]}")
            if i < len(all_images):
                print(f"Associated image: {all_images[i]}")
            print("Embedding in FAISS index:", faiss_index.reconstruct(i))
            print("-" * 50)

# Function to perform similarity search across all documents
def similarity_search(query, faiss_index, text_chunks):
    query_embedding = get_embeddings(query).reshape(1, -1).astype(np.float32)  # Correct dtype
    distances, indices = faiss_index.search(query_embedding, k=2)  # Use a sensible k value
    results = [text_chunks[i] for i in indices[0] if i < len(text_chunks)]  # Ensure index bounds
    return results

# Function to generate answer
def generate_answer(query, context):
    response = client.chat.completions.create(
        model="gpt-4o-2",
        messages=[
            {"role": "user", "content": f"Using the following context, answer the question: '{query}'\n\nContext: {context}"}
        ],
        temperature=0.7,
        max_tokens=150
    )
    answer = response.choices[0].message.content
    return answer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdfs():
    global all_text_chunks, faiss_index, all_images
    if 'files' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    files = request.files.getlist('files')
    pdf_paths = []

    # Save each uploaded PDF
    for file in files:
        if file.filename == '':
            continue
        pdf_path = os.path.join("upload", file.filename)
        file.save(pdf_path)
        pdf_paths.append(pdf_path)

    # Process each PDF and setup the index
    setup_pdf_index(pdf_paths)

    return jsonify({'message': 'PDFs uploaded and processed successfully.', 'images': all_images})

@app.route('/chat', methods=['POST'])
def chat_with_pdfs():
    global all_text_chunks, faiss_index, all_images
    data = request.get_json()
    query = data.get('query', '')

    if not query or not all_text_chunks or faiss_index is None:
        return jsonify({'answer': 'Please upload one or more PDFs first.'})

    results = similarity_search(query, faiss_index, all_text_chunks)
    context = "\n".join(results)
    answer = generate_answer(query, context)

    # Filter images based on query relevance
    relevant_images = []
    if "image" in query.lower() or "show me" in query.lower() or "display" in query.lower():
        relevant_images = [f"extracted_images/{img}" for img in all_images]

    return jsonify({'answer': answer, 'image_paths': relevant_images})

# Route to serve static images
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
