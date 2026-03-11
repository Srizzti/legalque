import os
import sqlite3
import PyPDF2
from sentence_transformers import SentenceTransformer

pdf_folder_path =  # Replace with your PDF folder path
output_db_path =  # SQLite database path

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdfs(pdf_folder):
    text_data = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith('.PDF'):
            pdf_path = os.path.join(pdf_folder, filename)
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text() + ' '  
                text_data.append((filename, text.strip()))  
    return text_data

def create_vector_database(text_data):
    documents = [text for _, text in text_data]

    vectors = model.encode(documents)  
    return vectors

def save_to_sqlite(filenames, vectors, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            filename TEXT NOT NULL,
            vector BLOB NOT NULL
        )
    ''')
    
    for filename, vector in zip(filenames, vectors):
        vector_blob = vector.tobytes()
        cursor.execute('INSERT INTO documents (filename, vector) VALUES (?, ?)', (filename, vector_blob))
    conn.commit()
    conn.close()


text_data = extract_text_from_pdfs(pdf_folder_path)
print(text_data)
vectors = create_vector_database(text_data)
print(vectors)
filenames = [filename for filename, _ in text_data]
save_to_sqlite(filenames, vectors, output_db_path)

print("Vector database created and saved successfully as SQLite.")
