import pdfplumber
import pandas as pd
import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_path
from processing.text_analysis import extract_keywords, summarize_text
from processing.preprocessing import clean_text

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pdf_path = "uploads/sample.pdf"

print("===== TEXT EXTRACTION =====")

all_text = ""   # store all text here

with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages):

        text = page.extract_text()

        if text:
            print(f"\nPage {i+1} Text:\n")
            print(text)

            all_text += text   # collect text for AI processing

        else:
            print(f"\nPage {i+1}: No text detected")

print("\n===== TABLE EXTRACTION =====")

tables_list = []

with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:

        tables = page.extract_tables()

        for table in tables:

            df = pd.DataFrame(table)
            tables_list.append(df)

            print("\nTable Found:\n")
            print(df)

# Save tables
for i, table in enumerate(tables_list):
    table.to_csv(f"outputs/table_{i+1}.csv", index=False)

print("\nTables saved in outputs folder")

print("\n===== OCR FOR SCANNED PDF =====")

images = convert_from_path(
    pdf_path,
    poppler_path=r"C:\Users\barul\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"
)

for i, image in enumerate(images):

    img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

    text = pytesseract.image_to_string(img)

    if text.strip() != "":
        print(f"\nOCR Page {i+1}:\n")
        print(text)

        with open(f"outputs/ocr_page_{i+1}.txt", "w", encoding="utf-8") as f:
            f.write(text)

        all_text += text   # include OCR text also

print("\nOCR text saved in outputs folder")

# -----------------------------
# DATA CLEANING & PREPROCESSING
# -----------------------------

# Remove line breaks
all_text = all_text.replace("\n", " ")

# Apply preprocessing
all_text = clean_text(all_text)

print("\n===== CLEANED TEXT =====")
print(all_text[:500])   # show first 500 characters

# -----------------------------
# NLP PROCESSING
# -----------------------------

print("\n===== KEYWORDS =====")
keywords = extract_keywords(all_text)
print(keywords)

print("\n===== SUMMARY =====")
summary = summarize_text(all_text)
print(summary)
