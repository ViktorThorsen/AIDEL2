import streamlit as st
from pypdf import PdfReader
from google import genai

client = genai.Client(api_key="API_NYCKEL")

st.title("Snabb PDF-koll (Gemini 2.0 Flash)")

file = st.file_uploader("Ladda upp en PDF", type=["pdf"])

if file:
    reader = PdfReader(file)
    pdf_text = ""
    for page in reader.pages:
        pdf_text += page.extract_text()

    st.success("PDF inladdad!")
    query = st.text_input("Vad undrar du?")

    if query:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config={
                "system_instruction": "Använd endast följande text för att svara på frågor."
            },
            contents=f"Kontext: {pdf_text}\n\nFråga: {query}"
        )
        
        st.write("**Svar:**")
        st.write(response.text)