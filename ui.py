# ui.py
import streamlit as st
import os
import pandas as pd
import zipfile
from io import BytesIO
from src.parser import parse_resume, extract_text_from_file
from src.report import generate_match_report, extract_skills
from src.utils import clear_folder
from src.bedrock_llm import set_bedrock_credentials  # New utility

# Folders
RESUME_UPLOAD_FOLDER = "data/resumes/"
JOB_DESC_UPLOAD_FOLDER = "data/job_descriptions/"
SELECTED_PROFILE_FOLDER = "selected_profile/"

os.makedirs(RESUME_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(JOB_DESC_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SELECTED_PROFILE_FOLDER, exist_ok=True)

# Function to reset application state
def reset_application():
    """Reset all application state and clear uploads."""
    # Clear session state variables
    for key in ["jd_skills", "match_reports", "selected_profiles", 
                "processed", "extracted_jd_skills", "jd_paths", "jd_text"]:
        if key in st.session_state:
            if isinstance(st.session_state[key], dict):
                st.session_state[key] = {}
            elif isinstance(st.session_state[key], list):
                st.session_state[key] = []
            elif isinstance(st.session_state[key], bool):
                st.session_state[key] = False
            else:
                st.session_state[key] = None
    
    # Clear file upload widgets by setting their keys to None
    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = 0
    st.session_state.file_uploader_key += 1
    
    # Clear folders
    clear_folder(RESUME_UPLOAD_FOLDER)
    clear_folder(JOB_DESC_UPLOAD_FOLDER)
    clear_folder(SELECTED_PROFILE_FOLDER)
    
    # Force page refresh
    st.experimental_rerun()

# Title
st.title("🔍 AI Profile Syncer")
st.write("Upload multiple Job Descriptions and Resumes. Supports .pdf, .docx, .txt, and .zip formats. Each resume will be matched against each JD using skill overlap and embedding similarity.")
st.divider()
st.sidebar.info("🚀 Powered by Capgemini")

# # Clear All Button in Sidebar (at the top)
# if st.sidebar.button("🧹 Clear All", help="Reset all fields and results"):
#     reset_application()

st.sidebar.header("🔐 AWS Credentials")

# === AWS API Key Inputs ===
with st.sidebar.expander("🔐 AWS Credentials (Required)"):
    st.session_state.aws_access_key = st.text_input("AWS Access Key ID", type="password")
    st.session_state.aws_secret_key = st.text_input("AWS Secret Access Key", type="password")
    st.session_state.aws_region = st.text_input("AWS Region", value="us-east-1")

# Initialize session states
if "jd_skills" not in st.session_state:
    st.session_state.jd_skills = {}
if "match_reports" not in st.session_state:
    st.session_state.match_reports = {}
if "selected_profiles" not in st.session_state:
    st.session_state.selected_profiles = {}
if "processed" not in st.session_state:
    st.session_state.processed = False
if "extracted_jd_skills" not in st.session_state:
    st.session_state.extracted_jd_skills = False
if "jd_paths" not in st.session_state:
    st.session_state.jd_paths = []
if "jd_text" not in st.session_state:
    st.session_state.jd_text = {}
if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0

# === Uploads ===
st.sidebar.header("📁 Upload Files")

# Helpers
def extract_files_from_zip(uploaded_zip, save_dir, allowed_ext):
    extracted = []
    with zipfile.ZipFile(BytesIO(uploaded_zip.read()), "r") as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.lower().endswith(tuple(allowed_ext)):
                full_path = os.path.join(save_dir, file_name)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "wb") as f:
                    f.write(zip_ref.read(file_name))
                extracted.append(full_path)
    return extracted

def save_file(uploaded_file, save_dir):
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return [file_path]

# Step 1: Upload JD
jd_input = st.sidebar.file_uploader(
    "Step 1: Upload Job Description(s) (TXT, PDF, DOCX or ZIP)",
    type=["txt", "pdf", "docx", "zip"],
    key=f"jd_uploader_{st.session_state.file_uploader_key}"
)

# Extract JD skills when JD is uploaded
extract_skills_button = None
if jd_input and not st.session_state.extracted_jd_skills:
    extract_skills_button = st.sidebar.button("Extract Skills from JD")

if extract_skills_button:
    if not (st.session_state.aws_access_key and st.session_state.aws_secret_key and st.session_state.aws_region):
        st.sidebar.error("❌ Please provide AWS credentials and region to extract skills.")
    else:
        # Set credentials dynamically
        set_bedrock_credentials(st.session_state.aws_access_key, st.session_state.aws_secret_key, st.session_state.aws_region)
        
        clear_folder(JOB_DESC_UPLOAD_FOLDER)
        st.session_state.jd_skills = {}
        
        with st.spinner(text="Extracting skills from job description..."):
            if jd_input.name.endswith(".zip"):
                st.session_state.jd_paths = extract_files_from_zip(jd_input, JOB_DESC_UPLOAD_FOLDER, ["txt", "pdf", "docx"])
            else:
                st.session_state.jd_paths = save_file(jd_input, JOB_DESC_UPLOAD_FOLDER)
                
            for jd_path in st.session_state.jd_paths:
                jd_text = extract_text_from_file(jd_path)
                jd_name = os.path.splitext(os.path.basename(jd_path))[0].replace(" ", "_")
                st.session_state.jd_text[jd_name] = jd_text
                
                # Extract skills from JD
                skills = extract_skills(jd_text, 
                                       st.session_state.aws_access_key, 
                                       st.session_state.aws_secret_key, 
                                       st.session_state.aws_region)
                st.session_state.jd_skills[jd_name] = {skill: {"selected": True, "weight": 1.0} for skill in skills}
            
            st.session_state.extracted_jd_skills = True

# Display skills selection after extraction
if st.session_state.extracted_jd_skills:
    st.sidebar.subheader("Step 2: Select Required Skills")
    
    # Display each JD's skills with selection and weightage
    for jd_name, skills_dict in st.session_state.jd_skills.items():
        with st.sidebar.expander(f"Skills from {jd_name}"):
            st.write(f"Select required skills and assign weightage (1.0 = normal weight)")
            
            # First display all skill selections with checkboxes
            for skill in sorted(skills_dict.keys()):
                skills_dict[skill]["selected"] = st.checkbox(
                    f"{skill}", 
                    value=skills_dict[skill]["selected"],
                    key=f"skill_{jd_name}_{skill}"
                )
            
            st.write("---")
            st.write("📊 **Optional: Add Weightage to Skills**")
            
            # Then display weightage inputs for selected skills
            for skill in sorted(skills_dict.keys()):
                if skills_dict[skill]["selected"]:
                    skills_dict[skill]["weight"] = st.slider(
                        f"Weight for {skill}", 
                        min_value=0.1, 
                        max_value=5.0, 
                        value=skills_dict[skill]["weight"],
                        step=0.1,
                        key=f"weight_{jd_name}_{skill}"
                    )

    # Step 3: Upload Resumes (only show this after skills are extracted)
    resume_input = st.sidebar.file_uploader(
        "Step 3: Upload Resumes (PDF, DOCX or ZIP)",
        type=["pdf", "docx", "zip"],
        accept_multiple_files=False,
        key=f"resume_uploader_{st.session_state.file_uploader_key}"
    )

    min_match_score = st.sidebar.number_input("Minimum Resume Match Score (%)", min_value=0, max_value=100, value=70)

    # Step 4: Process
    process_button = st.sidebar.button("Step 4: Process Matching")
    
    # Processing logic
    if process_button:
        if not (st.session_state.aws_access_key and st.session_state.aws_secret_key and st.session_state.aws_region):
            st.error("❌ Please provide AWS credentials and region to proceed.")
        elif not resume_input:
            st.error("❌ Please upload resume files to proceed.")
        else:
            # Set credentials dynamically
            set_bedrock_credentials(st.session_state.aws_access_key, st.session_state.aws_secret_key, st.session_state.aws_region)

            clear_folder(RESUME_UPLOAD_FOLDER)
            clear_folder(SELECTED_PROFILE_FOLDER)
            st.session_state.match_reports.clear()
            st.session_state.selected_profiles.clear()
            st.session_state.processed = False

            resume_paths = []

            with st.spinner(text="Matching in progress..."):
                if resume_input:
                    if resume_input.name.endswith(".zip"):
                        resume_paths = extract_files_from_zip(resume_input, RESUME_UPLOAD_FOLDER, ["pdf", "docx"])
                    else:
                        resume_paths = save_file(resume_input, RESUME_UPLOAD_FOLDER)

                if not resume_paths:
                    st.warning("Please upload resumes to proceed!")
                else:
                    resume_texts = {
                        os.path.basename(path): parse_resume(path)
                        for path in resume_paths
                    }

                    for jd_name, skills_data in st.session_state.jd_skills.items():
                        jd_text = st.session_state.jd_text[jd_name]
                        
                        # Filter selected skills and their weights
                        selected_skills = {skill: data["weight"] for skill, data in skills_data.items() if data["selected"]}
                        
                        if not selected_skills:
                            st.warning(f"No skills selected for JD: {jd_name}. Skipping this JD.")
                            continue
                        
                        results = generate_match_report(
                            resume_texts, 
                            jd_text, 
                            selected_skills=selected_skills,
                            aws_access_key=st.session_state.aws_access_key,
                            aws_secret_key=st.session_state.aws_secret_key,
                            aws_region=st.session_state.aws_region
                        )

                        selected_resumes = []
                        detailed_data = []

                        for result in results:
                            if result["embedding_score"] >= min_match_score:
                                selected_resumes.append(result["resume"])
                                src_path = os.path.join(RESUME_UPLOAD_FOLDER, result["resume"])
                                dst_path = os.path.join(SELECTED_PROFILE_FOLDER, result["resume"])
                                if os.path.exists(src_path):
                                    with open(src_path, "rb") as src, open(dst_path, "wb") as dst:
                                        dst.write(src.read())

                            detailed_data.append({
                                "Resume": result["resume"],
                                "Skills Match (%)": result["match_score"],
                                "Resume Match (%)": result["embedding_score"],
                                "All Resume Skills": ", ".join(sorted(result["all_resume_skills"])),
                                "Matching Skills with JD": ", ".join(sorted(result["matched_skills"])),
                                "Missing Skills from JD": ", ".join(sorted(result["missing_skills"])),
                                "Skill Weights Applied": result["skill_weights"] if "skill_weights" in result else "N/A"
                            })

                        st.session_state.match_reports[jd_name] = pd.DataFrame(detailed_data)
                        st.session_state.selected_profiles[jd_name] = selected_resumes
                        st.session_state.processed = True

# Display Results
if st.session_state.processed:
    st.header("Results")
    
    # Add a clear results button in the main area too
    if st.sidebar.button("🧹 Clear Results", help="Clear all results and start over"):
        reset_application()
    
    for jd_name, match_df in st.session_state.match_reports.items():
        st.subheader(f"📊 Matching Report for JD: {jd_name}")
        st.dataframe(match_df)

        selected = st.session_state.selected_profiles.get(jd_name, [])
        if selected:
            st.success(f"✅ {len(selected)} resumes matched for JD '{jd_name}' and saved.")
        else:
            st.warning(f"⚠️ No resumes matched the minimum score for JD '{jd_name}'.")

        st.download_button(
            f"Download CSV for {jd_name}",
            data=match_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{jd_name}_report.csv",
            mime="text/csv",
            key=f"csv_{jd_name}"
        )

        if selected:
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                for filename in selected:
                    filepath = os.path.join(SELECTED_PROFILE_FOLDER, filename)
                    if os.path.exists(filepath):
                        with open(filepath, "rb") as f:
                            zipf.writestr(filename, f.read())
            zip_buffer.seek(0)
            st.download_button(
                label=f"Download Matched Resumes (ZIP) for {jd_name}",
                data=zip_buffer,
                file_name=f"{jd_name}_selected_resumes.zip",
                mime="application/zip",
                key=f"zip_{jd_name}"
            )