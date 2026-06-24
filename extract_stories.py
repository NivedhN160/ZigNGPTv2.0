import os
from pypdf import PdfReader

stories_dir = 'stories'
output_file = 'stories_corpus.txt'

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def main():
    if not os.path.exists(stories_dir):
        print(f"Directory {stories_dir} not found.")
        return

    corpus_text = ""
    pdf_files = [f for f in os.listdir(stories_dir) if f.endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files in {stories_dir}.")

    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")
        pdf_path = os.path.join(stories_dir, pdf_file)
        text = extract_text_from_pdf(pdf_path)
        corpus_text += f"\n\n--- {pdf_file} ---\n\n"
        corpus_text += text

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(corpus_text)
    
    print(f"Successfully extracted {len(corpus_text)} characters to {output_file}.")

if __name__ == "__main__":
    main()
