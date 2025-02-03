import streamlit as st
from jamaibase import JamAI, protocol as p
import time
import os

# Initialize session state variables
def initialize_session_state():
    defaults = {
        "unique_time": time.time(),
        "knowledge_base_exist": False,
        "chat_history": [],
        "api_key": "",
        "project_id": "",
        "model": "ellm/Qwen/Qwen2-7B-Instruct",
        "k": 5,
        "temperature": 0.3,
        "top_p": 0.7,
        "max_tokens": 496,
        "knowledge_table_id": None,
        "logged_in": False  # Add logged_in status
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Function to clear credentials
def clear_credentials():
    st.session_state.api_key = ""
    st.session_state.project_id = ""
    st.session_state.unique_time = time.time()  # reset the unique time
    st.session_state.logged_in = False  # reset logged in status

# Multi-query refinement for deeper analysis
def refine_question(question):
    return [
        f"What medical conditions could be associated with: {question}?",
        f"What lab tests or imaging scans are recommended for: {question}?",
        f"What are the treatment options for conditions related to: {question}?"
    ]

# Function to ask a question using JamAI chat table
def ask_question(question):
    jamai = JamAI(api_key=st.session_state.api_key, project_id=st.session_state.project_id)
    table_id = f"chat-rag-{st.session_state.unique_time}"
    
    refined_questions = refine_question(question)
    responses = []

    for refined_q in refined_questions:
        try:
            completion = jamai.table.add_table_rows(
                table_type=p.TableType.chat,
                request=p.RowAddRequest(
                    table_id=table_id,
                    data=[dict(User=refined_q)],
                    stream=True,
                ),
            )
            full_response = ""
            for chunk in completion:
                if chunk.output_column_name == "AI":
                    full_response += chunk.text
            responses.append(full_response)
        except Exception as e:
            responses.append(f"Error fetching response: {e}")

    return "\n\n".join(responses)  # Returns a well-rounded answer

# Function to create knowledge base
def create_knowledge_base(jamai, file_upload):
    try:
        with st.spinner("Creating Knowledge Base..."):
            knowledge_simple = f"knowledge-simple-{st.session_state.unique_time}"
            knowledge_table = jamai.create_knowledge_table(
                p.KnowledgeTableSchemaCreate(
                    id=knowledge_simple,
                    cols=[],  # Define the columns here if needed
                    embedding_model="ellm/BAAI/bge-m3",  # Embedding model used for semantic search
                )
            )
        
        st.success("Successfully created Knowledge Base")

        # Step 2: Save the uploaded PDF locally before uploading it to the knowledge base
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current working directory
        file_path = os.path.join(current_dir, file_upload.name)  # Full path for saving the file
        
        with open(file_path, "wb") as f:
            f.write(file_upload.read())  # Write the uploaded file content to the file system
        
        # Step 3: Upload the file to the JamAI knowledge base
        with st.spinner("Uploading PDF to Knowledge Base..."):
            response = jamai.upload_file(
                p.FileUploadRequest(
                    file_path=file_path,  # Path to the file to be uploaded
                    table_id=knowledge_simple,  # The knowledge table where the file will be associated
                )
            )
            if response.ok:
                st.success("Successfully uploaded PDF to Knowledge Base!")
            else:
                st.error("File upload failed. Please try again.")
                return None

        # Step 4: Remove the file locally after upload
        os.remove(file_path)

        # Update the session state to reflect that the knowledge base exists
        st.session_state.knowledge_base_exist = True
        st.session_state.knowledge_table_id = knowledge_simple

        return knowledge_simple

    except Exception as e:
        st.warning(f"An error occurred during the knowledge base creation: {e}")
        # Clear sensitive session state in case of error
        clear_credentials()
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
                                messages=[ 
                                    p.ChatEntry.system(
                                        "You are an advanced medical AI specializing in clinical report analysis. "
                                        "Provide detailed explanations, potential diagnoses, statistical probabilities, "
                                        "comparisons with known medical conditions, and possible follow-up tests or treatments. "
                                        "Cite sources where relevant and summarize key takeaways concisely."
                                    )
                                ],
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
        st.warning(f"An error occurred while creating the chat table: {e}")
        clear_credentials()

# Main app function
def main():
    st.title("MedLLM: LLM Chatbot for Medical Reports Analysis")

    initialize_session_state()

    # Sidebar for configuration
    with st.sidebar:
        st.image("additional_resources/Jamai-Long-Black-Main.icuEAbYB.svg")
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
            background-color: #FF0000;
            border-radius: 5px;
            display: block;
            text-align: center;
            ">
            🔑 Login to JamAI
            </a>
            """,
            unsafe_allow_html=True
        )
        st.header("🔧 Settings")
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
                if not st.session_state.knowledge_table_id:
                    st.session_state.knowledge_table_id = create_knowledge_base(jamai, file_upload)
                if st.session_state.knowledge_table_id:
                    create_chat_table(jamai, st.session_state.knowledge_table_id)

        with st.expander("Advanced Settings"):
            st.session_state.k = st.slider("k (retrieval depth)", 1, 20, 5)
            st.session_state.max_tokens = st.slider("Max tokens", 96, 960, 496, step=8)
            st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.3, step=0.05)
            st.session_state.top_p = st.slider("Top-P", 0.0, 1.0, 0.7, step=0.05)

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
