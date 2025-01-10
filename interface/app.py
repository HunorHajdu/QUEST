import streamlit as st
import tempfile
import os

def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'files_processed' not in st.session_state:
        st.session_state.files_processed = False

def display_chat_message(role, content):
    with st.chat_message(role):
        st.write(content)

def main():
    st.set_page_config(page_title="Q.U.E.S.T.", layout="wide")
    initialize_session_state()
    
    st.title("📚 Q.U.E.S.T.")
    
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload your PDF documents",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        st.header("Instructions")
        st.markdown("""
        1. Upload your PDF documents
        2. Wait for processing to complete
        3. Start chatting!
        """)
        
        if uploaded_files and not st.session_state.files_processed:
            with st.spinner("Processing documents..."):
                with tempfile.TemporaryDirectory() as temp_dir:
                    pdf_files = []
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        pdf_files.append(file_path)
                    
                    # Add the processing of the PDF file here
                    
                    st.session_state.files_processed = True
                    st.success(f"✅ {len(uploaded_files)} files processed!")
    
    if st.session_state.files_processed:
        for message in st.session_state.messages:
            display_chat_message(message["role"], message["content"])
        
        if prompt := st.chat_input("Ask a question about your documents"):
            display_chat_message("user", prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = "Replace this."
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
    
    else:
        st.info("👆 Please upload your PDF documents in the sidebar to get started")
        
        if not st.session_state.messages:
            display_chat_message("assistant", "Hello! I'm your PDF assistant. Please upload some documents to get started.")

    if st.session_state.messages:
        if st.sidebar.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()