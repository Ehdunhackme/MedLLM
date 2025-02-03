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
        "token": "",  
        "project_id": "",
        "model": "ellm/Qwen/Qwen2-7B-Instruct",  # Default model, can be changed later
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
    st.session_state.token = ""  
    st.session_state.project_id = ""
    st.session_state.unique_time = time.time()  
    st.session_state.logged_in = False  

# Function to create chat table
def create_chat_table(jamai, knowledge_simple):
    try:
        with st.spinner("Creating Chat Table..."):
            table = jamai.create_chat_table(
                p.ChatTableSchemaCreate(
                    id=f"chat-rag-{st.session_state.unique_time}",
                    cols=[
                        p.ColumnSchemaCreate(id="User", dtype="str"),
                        p.ColumnSchemaCreate(
                            id="AI",
                            dtype="str",
                            gen_config=p.ChatRequest(
                                model=st.session_state.model,  
                                messages=[p.ChatEntry.system(
                                    "You are an advanced medical AI specializing in clinical report analysis. "
                                    "You have access to a vast database of medical knowledge. "
                                    "When analyzing clinical reports, provide a detailed and well-reasoned explanation for each finding, including the following: "
                                    "- Comprehensive explanations of medical conditions or symptoms mentioned in the report. "
                                    "- A thorough comparison with related medical conditions, including differences and similarities. "
                                    "- Statistical probabilities or likelihood of diagnoses where applicable, based on available data and medical literature. "
                                    "- Recommendations for additional lab tests, imaging scans, or other diagnostic procedures based on the findings. "
                                    "- A summary of potential treatments, interventions, and follow-up care for the patient. "
                                    "- Cite reliable medical sources, studies, or clinical guidelines where relevant to support your recommendations. "
                                    "Ensure that your analysis is structured, clear, and concise, and that each step of the reasoning is explained thoroughly."
                                )]
                                rag_params=p.RAGParams(
                                    table_id=knowledge_simple,
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
        clear_credentials()
        st.warning("An error occurred. Please check your credentials and try again.")

# Function to ask a question with improved streaming output
def ask_question(question):
    jamai = JamAI(api_key=st.session_state.token, project_id=st.session_state.project_id)
    
    completion = jamai.add_table_rows(
        "chat",
        p.RowAddRequest(
            table_id=f"chat-rag-{st.session_state.unique_time}",
            data=[dict(User=question)],
            stream=True,
        ),
    )
    
    full_response = ""

    for chunk in completion:
        if chunk.output_column_name != "AI":
            continue
        if isinstance(chunk, p.GenTableStreamReferences):
            pass
        else:
            full_response += chunk.text
            yield full_response

# Function to create knowledge base
def create_knowledge_base(jamai, file_upload):
    try:
        with st.spinner("Creating Knowledge Base..."):
            knowledge_simple = f"knowledge-simple-{st.session_state.unique_time}"
            knowledge_table = jamai.create_knowledge_table(
                p.KnowledgeTableSchemaCreate(
                    id=knowledge_simple,
                    cols=[],  
                    embedding_model="ellm/BAAI/bge-m3",  
                )
            )
        
        st.success("Successfully created Knowledge Base")

        # Step 2: Save the uploaded PDF locally before uploading it to the knowledge base
        current_dir = os.path.dirname(os.path.abspath(__file__)) 
        file_path = os.path.join(current_dir, file_upload.name)  
        
        with open(file_path, "wb") as f:
            f.write(file_upload.read())  
        
        # Step 3: Upload the file to the JamAI knowledge base
        with st.spinner("Uploading PDF to Knowledge Base..."):
            response = jamai.upload_file(
                p.FileUploadRequest(
                    file_path=file_path,  
                    table_id=knowledge_simple,  
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
        clear_credentials()
        return None

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
            background-color: #FF0000;  /* Red color */
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
        token = st.text_input('🔑 JamAI Token', type='password', value=st.session_state.token)  # Changed to token
        project_id = st.text_input('📌 Project ID', value=st.session_state.project_id)

        if token and project_id:
            st.session_state.token = token
            st.session_state.project_id = project_id
            st.session_state.logged_in = True
            st.success(f"✅ Logged in as **{st.session_state.project_id}**")
        else:
            st.warning("🔒 Please enter your Token and Project ID.")

        # Upload Medical Report Button
        file_upload = st.file_uploader("📂 Upload Medical Report", type=["pdf"])
        if st.button("Upload Medical Report"):
            if not st.session_state.get("logged_in"):
                st.warning("🔒 Please log in first.")
            elif not file_upload:
                st.warning("⚠️ Please upload a PDF file.")
            else:
                jamai = JamAI(api_key=st.session_state.token, project_id=st.session_state.project_id)  # Use token
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
