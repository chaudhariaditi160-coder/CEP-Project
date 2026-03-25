from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz
import os
from transformers import pipeline

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
IMAGE_FOLDER = "images"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# ✅ Use supported pipeline
summarizer = pipeline("text-generation", model="gpt2")
qa_pipeline = pipeline("text-generation", model="gpt2")

document_text = ""

# 🔹 Function to handle long text
def generate_summary(text):
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    final_summary = ""

    for chunk in chunks[:3]:  # limit chunks for speed
        prompt = "Summarize the following text:\n" + chunk + "\nSummary:"
        
        result = summarizer(
            prompt,
            max_length=150,
            do_sample=False
        )

        summary = result[0]["generated_text"].replace(prompt, "")
        final_summary += summary.strip() + " "

    return final_summary.strip()


@app.route("/")
def home():
    return "Backend is running 🚀"


@app.route("/upload", methods=["POST"])
def upload():
    global document_text
    document_text = ""

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    pdf = fitz.open(file_path)

    extracted_text = ""
    image_names = []

    for page_index in range(len(pdf)):
        page = pdf[page_index]

        # ✅ Extract text
        extracted_text += page.get_text()

        # ✅ Extract images
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf.extract_image(xref)

            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image_name = f"image_{page_index}_{img_index}.{image_ext}"
            image_path = os.path.join(IMAGE_FOLDER, image_name)

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            image_names.append(image_name)

    document_text = extracted_text

    # ✅ Generate summary
    summary = generate_summary(document_text)

    return jsonify({
        "summary": summary,
        "images": image_names
    })


@app.route("/ask", methods=["POST"])
def ask():
    global document_text

    if document_text == "":
        return jsonify({"answer": "Upload PDF first"})

    data = request.get_json()
    question = data.get("question", "")

    if question == "":
        return jsonify({"answer": "No question provided"})

    # ✅ Updated QA logic for GPT-2
    prompt = f"Context: {document_text[:1500]}\nQuestion: {question}\nAnswer:"

    result = qa_pipeline(
        prompt,
        max_length=200,
        do_sample=False
    )

    answer = result[0]["generated_text"].replace(prompt, "")

    return jsonify({
        "answer": answer.strip()
    })


if __name__ == "__main__":
    app.run(debug=True)