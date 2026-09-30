import PyPDF2
from transformers import pipeline

# Load summarizer model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def simplify_legal_text(legal_text):
    # Chunk text because model has limit
    chunks = [legal_text[i:i+1000] for i in range(0, len(legal_text), 1000)]
    simplified = ""
    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=40, do_sample=False)
        simplified += summary[0]['summary_text'] + "\n\n"
    return simplified

if __name__ == "__main__":
    pdf_path = "LegalEase.pdf"
    print("Reading PDF...")
    raw_text = extract_text_from_pdf(pdf_path)

    print("Simplifying...")
    result = simplify_legal_text(raw_text)

    print("\n--- LegalEase Simplified Output ---\n")
    print(result)

    # Save output
    with open("simplified_output.txt", "w", encoding="utf-8") as f:
        f.write(result)
