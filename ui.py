# ui.py
import streamlit as st
import os
import pandas as pd
import zipfile
from io import BytesIO
from src.parser import parse_resume, extract_text_from_file
from src.report import generate_match_report
from src.utils import clear_folder
from src.bedrock_llm import set_bedrock_credentials  # New utility
import unicodedata

# Folders
#
RESUME_UPLOAD_FOLDER = "data/resumes/"
JOB_DESC_UPLOAD_FOLDER = "data/job_descriptions/"
SELECTED_PROFILE_FOLDER = "selected_profile/"

os.makedirs(RESUME_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(JOB_DESC_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SELECTED_PROFILE_FOLDER, exist_ok=True)

# Title
st.title("🔍 AI Profile Syncer")
st.write("Upload multiple Job Descriptions and Resumes. Supports .pdf, .docx, .txt, and .zip formats. Each resume will be matched against each JD using skill overlap and embedding similarity.")
st.divider()
st.sidebar.info("🚀 Powered by Capgemini")
st.sidebar.header("🔐 AWS Credentials")

# === NEW: AWS API Key Inputs ===
with st.sidebar.expander("🔐 AWS Credentials (Required)"):
    st.session_state.aws_access_key = st.text_input("AWS Access Key ID", type="password")
    st.session_state.aws_secret_key = st.text_input("AWS Secret Access Key", type="password")
    st.session_state.aws_region = st.text_input("AWS Region", value="us-east-1")

#Clear folder
if "folders_cleared" not in st.session_state:
            clear_folder(RESUME_UPLOAD_FOLDER)
            clear_folder(JOB_DESC_UPLOAD_FOLDER)
            st.session_state.folders_cleared = True
            
# === Uploads ===
st.sidebar.header("📁 Upload Files")

jd_input = st.sidebar.file_uploader(
    "Upload Job Description(s) (TXT, PDF, DOCX or ZIP)",
    type=["txt", "pdf", "docx", "zip"]
)

resume_input = st.sidebar.file_uploader(
    "Upload Resumes (PDF, DOCX or ZIP)",
    type=["pdf", "docx", "zip"],
    accept_multiple_files=False
)

min_match_score = st.sidebar.number_input("Minimum Resume Match Score (%)", min_value=0, max_value=100, value=70)

# Session state
if "match_reports" not in st.session_state:
    st.session_state.match_reports = {}
if "selected_profiles" not in st.session_state:
    st.session_state.selected_profiles = {}
if "processed" not in st.session_state:
    st.session_state.processed = False

# Helpers
def extract_files_from_zip(uploaded_zip, save_dir, allowed_ext):
    extracted = []
    with zipfile.ZipFile(BytesIO(uploaded_zip.read()), "r") as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.lower().endswith(tuple(allowed_ext)):
                # Normalize file name (remove special chars and spaces)
                safe_name = unicodedata.normalize("NFKD", os.path.basename(file_name)).encode("ascii", "ignore").decode("ascii")
                safe_name = safe_name.replace(" ", "_")
                full_path = os.path.join(save_dir, safe_name)

                with open(full_path, "wb") as f:
                    f.write(zip_ref.read(file_name))
                extracted.append(full_path)
    return extracted


def save_file(uploaded_file, save_dir):
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return [file_path]

# Pre-process JD file and populate dropdown
jd_paths = []
if jd_input:
    if jd_input.name.endswith(".zip"):
        jd_paths = extract_files_from_zip(jd_input, JOB_DESC_UPLOAD_FOLDER, ["txt", "pdf", "docx"])
    else:
        jd_paths = save_file(jd_input, JOB_DESC_UPLOAD_FOLDER)

jd_file_map = {os.path.basename(path): path for path in jd_paths}
selected_jd_name = None
selected_jd_path = None

if jd_file_map:
    selected_jd_name = st.sidebar.selectbox("Select Job Description to Match", list(jd_file_map.keys()))
    selected_jd_path = jd_file_map[selected_jd_name]

# Processing logic
if st.sidebar.button("Process Matching"):
    if not (st.session_state.aws_access_key and st.session_state.aws_secret_key and st.session_state.aws_region):
        st.error("❌ Please provide AWS credentials and region to proceed.")
    else:
        set_bedrock_credentials(st.session_state.aws_access_key, st.session_state.aws_secret_key, st.session_state.aws_region)

        clear_folder(SELECTED_PROFILE_FOLDER)
        st.session_state.match_reports.clear()
        st.session_state.selected_profiles.clear()
        st.session_state.processed = False

        jd_paths = [selected_jd_path] if selected_jd_path else []
        resume_paths = []

        with st.spinner(text="Matching in progress..."):
            if resume_input:
                if resume_input.name.endswith(".zip"):
                    resume_paths = extract_files_from_zip(resume_input, RESUME_UPLOAD_FOLDER, ["pdf", "docx"])
                else:
                    resume_paths = save_file(resume_input, RESUME_UPLOAD_FOLDER)

            if not jd_paths or not resume_paths:
                st.warning("Please upload both job descriptions and resumes!")
            else:
                resume_texts = {
                    os.path.basename(path): parse_resume(path)
                    for path in resume_paths
                }

                for jd_path in jd_paths:
                    jd_text = extract_text_from_file(jd_path)
                    results = generate_match_report(
                        resume_texts,
                        jd_text,
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
                            "Name": result["name"],
                            "Email": result["email"],
                            "Location": result["location"],
                            "Years of Experience": result["years of experience"],
                            "Skills Match (%)": result["match_score"],
                            "Resume Match (%)": result["embedding_score"],
                            "All Resume Skills": ", ".join(sorted(result["all_resume_skills"])),
                            "Matching Skills with JD": ", ".join(sorted(result["matched_skills"])),
                            "Missing Skills from JD": ", ".join(sorted(result["missing_skills"]))
                        })

                    jd_name = os.path.splitext(os.path.basename(jd_path))[0].replace(" ", "_")
                    st.session_state.match_reports[jd_name] = pd.DataFrame(detailed_data)
                    st.session_state.selected_profiles[jd_name] = selected_resumes
                    st.session_state.processed = True

# Display Results
if st.session_state.processed:
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
