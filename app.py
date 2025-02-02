import streamlit as st
from jamaibase import JamAI, protocol as p
import time
import os

# Initialize session state variables
def initialize_session_state():
    if "unique_time" not in st.session_state:
        st.session_state.unique_time = time.time()
    if "knowledge_base_exist" not in st.session_state:
        st.session_state.knowledge_base_exist = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "api_key" not in st.session_state:
        st.session_state.api_key = st.session_state.get("api_key", "")
    if "project_id" not in st.session_state:
        st.session_state.project_id = st.session_state.get("project_id", "")
    if "model" not in st.session_state:
        st.session_state.model = "ellm/Qwen/Qwen2-7B-Instruct"
    if "k" not in st.session_state:
        st.session_state.k = 2
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.01
    if "top_p" not in st.session_state:
        st.session_state.top_p = 0.01
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 496

# Function to ask a question using JamAI chat table
def ask_question(question):
    jamai = JamAI(api_key=st.session_state.api_key, project_id=st.session_state.project_id)
    
    # Ensure chat table exists
    table_id = f"chat-rag-{st.session_state.unique_time}"
    
    # Add user input to the chat table and stream the response
    completion = jamai.table.add_table_rows(
        table_type=p.TableType.chat,
        request=p.RowAddRequest(
            table_id=table_id,
            data=[dict(User=question)],
            stream=True,
        ),
    )

    full_response = ""
    for chunk in completion:
        if chunk.output_column_name == "AI":
            full_response += chunk.text
            yield full_response  # Yielding response in chunks for streaming effect

# Function to create knowledge base and upload medical reports
def create_knowledge_base(jamai, file_upload):
    try:
        with st.spinner("Creating Knowledge Base..."):
            knowledge_table_id = f"knowledge-table-{st.session_state.unique_time}"
            jamai.create_knowledge_table(
                p.KnowledgeTableSchemaCreate(
                    id=knowledge_table_id,
                    cols=[],
                    embedding_model="ellm/BAAI/bge-m3",
                )
            )

        st.success("Successfully created Knowledge Base")

        # Save uploaded PDF locally before uploading to JamAI
        file_path = os.path.join(os.getcwd(), file_upload.name)
        with open(file_path, "wb") as f:
            f.write(file_upload.read())

        with st.spinner("Uploading PDF to Knowledge Base..."):
            response = jamai.file.upload_file(file_path)
            assert response.ok

        st.success("Successfully uploaded PDF to Knowledge Base!")
        st.session_state.knowledge_base_exist = True
        st.session_state.knowledge_table_id = knowledge_table_id

        os.remove(file_path)  # Remove local copy after upload
        return knowledge_table_id

    except Exception as e:
        st.warning(f"An error occurred while uploading the PDF: {e}")
        return None

# Function to create chat table
def create_chat_table(jamai, knowledge_table_id):
    try:
        with st.spinner("Creating Chat Table..."):
            jamai.create_chat_table(
                p.ChatTableSchemaCreate(
                    id=f"chat-rag-{st.session_state.unique_time}",
                    cols=[
                        p.ColumnSchemaCreate(id="User", dtype="str"),
                        p.ColumnSchemaCreate(
                            id="AI",
                            dtype="str",
                            gen_config=p.ChatRequest(
                                model=st.session_state.model,
                                messages=[p.ChatEntry.system("You are a concise medical AI assistant.")],
                                rag_params=p.RAGParams(
                                    table_id=knowledge_table_id,
                                    k=st.session_state.k,
                                ),
                                temperature=st.session_state.temperature,
                                top_p=st.session_state.top_p,
                                max_tokens=st.session_state.max_tokens,
                            ).model_dump(),
                        ),
                    ],
                )
            )
        st.success("Successfully created Chat Table")
    except Exception as e:
        st.warning("An error occurred. Please check your credentials and try again.")

# Main app function
def main():
    st.title("MedLLM: LLM Chatbot for Medical Reports Analysis")

    initialize_session_state()

    # Sidebar for configuration
    with st.sidebar:
        st.image("additional_resources/Jamai-Long-Black-Main.icuEAbYB.svg", use_container_width=True)
        st.markdown(
            """
            <a href="https://cloud.jamaibase.com/" style="
            display: inline-block;
            padding: 10px 30px;
            font-size: 1.2rem;
            font-weight: 600;
            text-align: center;
            text-decoration: none;
            color: #ffffff;
            background-color: #007bff;
            border-radius: 5px;
            display: block;
            text-align: center;
            ">
            🔑 Login to JamAI
            </a>
            """,
            unsafe_allow_html=True
        )

        st.header("🔧 Configuration")
        api_key = st.text_input('🔑 JamAI API KEY', type='password', value=st.session_state.api_key)
        project_id = st.text_input('📌 Project ID', value=st.session_state.project_id)

        if api_key and project_id:
            st.session_state.api_key = api_key
            st.session_state.project_id = project_id
            st.session_state.logged_in = True
            st.success(f"✅ Logged in as **{st.session_state.project_id}**")
        else:
            st.warning("🔒 Please enter your API Key and Project ID.")

        # Upload Medical Report Button
        file_upload = st.file_uploader("📂 Upload Medical Report", type=["pdf"])
        if st.button("Upload Medical Report"):
            if not st.session_state.get("logged_in"):
                st.warning("🔒 Please log in first.")
            elif not file_upload:
                st.warning("⚠️ Please upload a PDF file.")
            else:
                jamai = JamAI(api_key=st.session_state.api_key, project_id=st.session_state.project_id)
                knowledge_table_id = create_knowledge_base(jamai, file_upload)
                if knowledge_table_id:
                    create_chat_table(jamai, knowledge_table_id)

    # Chat input for user questions
    if question := st.chat_input("Ask a medical question, describe symptoms, or query your report..."):
        if st.session_state.knowledge_base_exist:
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""

                for response in ask_question(question):
                    full_response += response
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)

            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
        else:
            st.warning("Please upload a medical report and create a Knowledge Base first.")

if __name__ == "__main__":
    main()
