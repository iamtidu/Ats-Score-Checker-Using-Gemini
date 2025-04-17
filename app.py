import streamlit as st
import google.generativeai as genai
import os
import fitz # PyMuPDF
import re
from dotenv import load_dotenv

# --- Configuration ---
MODEL_NAME = "gemini-1.5-flash" # Or another suitable model


# --- Load API Key ---
# Try loading from .env file for local development
load_dotenv()
# Use Streamlit secrets if available (preferred for deployment), otherwise use environment variable
try:
    # Attempt to get the key from Streamlit secrets first
    API_KEY = st.secrets["GEMINI_API_KEY"]
    # print("Loaded API Key from Streamlit Secrets")
except (KeyError, FileNotFoundError):
    # If secrets aren't found (e.g., local dev without secrets file), use environment variable
    API_KEY = os.getenv("GEMINI_API_KEY")
    # print("Loaded API Key from Environment Variable")

# --- Initialize Gemini Client ---
# We only configure it once. Error handling happens during API calls.
if API_KEY:
    genai.configure(api_key=API_KEY)
    # print("Gemini AI Configured")
else:
    # If no API key is found at all, display an error and stop
    st.error("🔴 Error: Gemini API Key not found. Please set it in Streamlit secrets or a .env file.")
    st.stop() # Stop execution if no API key

# --- Helper Functions ---

@st.cache_data(show_spinner=False) # Cache PDF text extraction
def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file."""
    if uploaded_file is None:
        return ""
    try:
        # Read bytes from uploaded file
        pdf_bytes = uploaded_file.getvalue()
        # Open PDF from bytes
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n\n" # Add space between pages
        doc.close()
        return full_text.strip()
    except Exception as e:
        st.error(f"Error extracting PDF text: {e}")
        return ""

def call_gemini_api(prompt):
    """Calls the Gemini API and handles potential errors."""
    if not API_KEY:
        st.error("Gemini API Key not configured.")
        return None
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        # Basic check for valid response structure (adjust based on API version)
        if response and hasattr(response, 'text'):
             return response.text
        elif response and response.prompt_feedback.block_reason:
             st.warning(f"API call blocked: {response.prompt_feedback.block_reason}")
             return f"*API call blocked: {response.prompt_feedback.block_reason}*"
        else:
             st.warning("Received an unexpected response structure from Gemini.")
             return "*Received no valid response from the AI.*"

    except Exception as e:
        st.error(f"Gemini API Error: {e}")
        # Provide more specific error details if possible
        if "API key not valid" in str(e):
             st.error("Please check if your Gemini API Key is correct and valid.")
        return None

def parse_ats_score(response_text):
    """Extracts the ATS score from the Gemini response."""
    if response_text is None:
        return 0, "Error receiving response."
    # Regex to find "Score: NUMBER" at the beginning of the string
    match = re.match(r"Score:\s*(\d+)", response_text)
    if match:
        score = int(match.group(1))
        explanation = response_text[len(match.group(0)):].strip()
        return min(100, max(0, score)), explanation # Clamp score 0-100
    else:
        st.warning("Could not parse ATS score from the response.")
        return 0, response_text # Return original text if score not found


# --- Initialize Session State ---
if "job_description" not in st.session_state:
    st.session_state.job_description = ""
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = ""
if "ats_score" not in st.session_state:
    st.session_state.ats_score = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False


# --- Streamlit App Layout ---

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

st.title("📄 AI Resume Analyzer powered by Gemini")
st.markdown("Analyze resumes against job descriptions, get insights, and chat about the results.")
st.divider()

# --- Input Section ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Job Description")
    st.session_state.job_description = st.text_area(
        "Paste the Job Description here:",
        value=st.session_state.job_description,
        height=300,
        key="jd_input" # Add key for stability if needed
    )

with col2:
    st.subheader("Upload Resume")
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF only):",
        type="pdf",
        key="pdf_uploader"
    )

    # Process PDF when uploaded
    if uploaded_file is not None:
        # Check if it's a new file before processing
        # Simple check based on name, could be more robust (e.g., check size/hash if needed)
        if not st.session_state.get("processed_file_name") or st.session_state.processed_file_name != uploaded_file.name:
            with st.spinner("Extracting text from PDF..."):
                extracted_text = extract_text_from_pdf(uploaded_file)
                if extracted_text:
                    st.session_state.resume_text = extracted_text
                    st.session_state.pdf_processed = True
                    st.session_state.processed_file_name = uploaded_file.name # Store name of processed file
                    # Clear previous analysis results when new PDF is uploaded
                    st.session_state.analysis_result = ""
                    st.session_state.ats_score = 0
                    st.session_state.chat_history = []
                    st.success(f"Successfully extracted text from '{uploaded_file.name}'")
                else:
                    st.session_state.pdf_processed = False
                    st.session_state.processed_file_name = None # Reset processed file name on error
        # Show message even if already processed
        st.info(f"Current resume: '{uploaded_file.name}'")

# --- Analysis Actions ---
if st.session_state.pdf_processed:
    st.divider()
    st.subheader("Analyze Resume")

    # Use columns for button layout
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

    with btn_col1:
        if st.button("📝 Summary", key="summary_btn"):
            if st.session_state.resume_text:
                with st.spinner("Generating Summary..."):
                    prompt = f"""Provide a concise summary (2-3 sentences) highlighting the key qualifications and experience from the following resume text:

Resume Text:
---
{st.session_state.resume_text}
---"""
                    result = call_gemini_api(prompt)
                    if result is not None:
                        st.session_state.analysis_result = result
                        st.session_state.ats_score = 0 # Reset score display
            else:
                st.warning("Please upload and process a resume first.")

    with btn_col2:
        if st.button("💡 Improvements", key="improve_btn"):
            if st.session_state.resume_text:
                 with st.spinner("Generating Improvement Suggestions..."):
                    prompt = f"""Analyze the following resume text and provide 3-5 specific, actionable suggestions for improvement. Format the suggestions clearly using bullet points. Focus on areas like skills presentation, quantifying achievements, clarity, and structure. Mention how these improvements could better align with typical job requirements.

Resume Text:
---
{st.session_state.resume_text}
---"""
                    result = call_gemini_api(prompt)
                    if result is not None:
                        st.session_state.analysis_result = result
                        st.session_state.ats_score = 0 # Reset score display
            else:
                st.warning("Please upload and process a resume first.")

    with btn_col3:
        if st.button("🔍 Missing Keywords", key="keywords_btn"):
            if st.session_state.resume_text and st.session_state.job_description:
                with st.spinner("Finding Missing Keywords..."):
                    prompt = f"""Compare the following resume text against the job description. Identify and list key skills, technologies, or qualifications mentioned in the job description that are either missing or significantly underrepresented in the resume. Format as a list.

Job Description:
---
{st.session_state.job_description}
---

Resume Text:
---
{st.session_state.resume_text}
---"""
                    result = call_gemini_api(prompt)
                    if result is not None:
                        st.session_state.analysis_result = result
                        st.session_state.ats_score = 0 # Reset score display
            elif not st.session_state.job_description:
                 st.warning("Please enter the Job Description first.")
            else:
                st.warning("Please upload and process a resume first.")

    with btn_col4:
        if st.button("📊 % Match (ATS)", key="match_btn"):
            if st.session_state.resume_text and st.session_state.job_description:
                with st.spinner("Calculating ATS Score..."):
                    prompt = f"""Analyze the alignment between the resume and job description. Consider keywords, skills, experience relevance. Provide an estimated percentage match score (integer 0-100).
**Output Format:** Start the response *only* with "Score: [percentage]" on the first line, followed by a brief explanation on subsequent lines. Example:
Score: 75
The resume shows good alignment...

Job Description:
---
{st.session_state.job_description}
---

Resume Text:
---
{st.session_state.resume_text}
---"""
                    result = call_gemini_api(prompt)
                    if result is not None:
                        score, explanation = parse_ats_score(result)
                        st.session_state.ats_score = score
                        st.session_state.analysis_result = f"**ATS Score Explanation:**\n\n{explanation}"
            elif not st.session_state.job_description:
                 st.warning("Please enter the Job Description first.")
            else:
                st.warning("Please upload and process a resume first.")

    st.divider()

    # --- Results Display ---
    res_col1, res_col2 = st.columns([3, 1]) # Give more space to analysis text

    with res_col1:
        st.subheader("Analysis / Explanation")
        # Use markdown for better formatting of the result
        st.markdown(
             f"""
             <div style="background-color:#000000; padding: 15px; border-radius: 8px; min-height: 200px; max-height: 400px; overflow-y: auto; border: 1px solid #d1d9e6;">
             {st.session_state.analysis_result or "Click an analysis button above..."}
             </div>
             """,
             unsafe_allow_html=True
         )


    with res_col2:
        st.subheader("Estimated ATS Score")
        st.metric(
            label="Match Score",
            value=f"{st.session_state.ats_score}%",
            delta=None # No delta comparison for now
        )
        st.caption("(Based on comparison with Job Description)")

    st.divider()

    # --- Chat Section ---
    st.subheader("💬 Chat with AI about the Resume")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"]) # Display message content as Markdown

    # Chat input
    if prompt := st.chat_input("Ask something about the resume..."):
        if not st.session_state.resume_text:
            st.warning("Please upload and process a resume before chatting.")
        else:
            # Add user message to history and display
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Construct prompt for AI
            # Include limited history to avoid overly long prompts
            history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history[-5:]]) # Include last 5 messages
            chat_prompt = f"""You are a helpful AI assistant analyzing a resume against a job description.
Current Job Description Context:
---
{st.session_state.job_description or "N/A"}
---
Full Resume Text:
---
{st.session_state.resume_text}
---
Previous Conversation (last few messages):
{history_context}
User Query: {prompt}

Provide a helpful and relevant response based on the resume and the conversation context, using Markdown formatting where appropriate (like lists or bold text)."""

            # Call API and display response
            with st.spinner("AI is thinking..."):
                 response = call_gemini_api(chat_prompt)
                 if response:
                     st.session_state.chat_history.append({"role": "assistant", "content": response})
                     with st.chat_message("assistant"):
                         st.markdown(response)
                 else:
                     # Error is handled within call_gemini_api and displayed via st.error
                     # Optionally add an error marker to chat history
                      st.session_state.chat_history.append({"role": "assistant", "content": "*Sorry, I encountered an error processing your request.*"})
                      with st.chat_message("assistant"):
                          st.markdown("*Sorry, I encountered an error processing your request.*")


else:
    st.info("Upload a PDF resume to get started.")


# --- Footer or additional info ---
st.divider()
st.caption("Powered by Google Gemini | Develop By Udit")