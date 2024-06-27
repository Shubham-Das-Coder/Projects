import streamlit as st
from PyPDF2 import PdfReader
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Load the model and the tokenizer
model_name = "human-centered-summarization/financial-summarization-pegasus"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

# Streamlit app
st.title("PDF Financial Summarizer")
st.write("Upload a PDF file to extract the text and summarize the financial content.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from PDF
    with st.spinner('Extracting text from PDF...'):
        text = extract_text_from_pdf(uploaded_file)
        st.success('Text extracted successfully!')
        st.write("Extracted Text:")
        st.write(text[:2000])  # Display first 2000 characters of the extracted text

    # Slider for summary length
    summary_length = st.slider("Select summary length (number of words)", 100, 2000, 200)

    if st.button('Generate Summary'):
        # Tokenize the text
        input_ids = tokenizer(text, return_tensors="pt").input_ids

        # Generate the summary
        with st.spinner('Generating summary...'):
            output = model.generate(
                input_ids, 
                max_length=summary_length, 
                num_beams=5, 
                early_stopping=True
            )
            summary = tokenizer.decode(output[0], skip_special_tokens=True)
            st.success('Summary generated successfully!')
            st.write("Summary:")
            st.write(summary)
