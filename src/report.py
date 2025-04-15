from src.bedrock_llm import run_skill_extraction_prompt
from src.matching import calculate_similarity
from src.logger import logger
import time
from src.matching import calculate_similarity

def extract_skills(text, aws_access_key, aws_secret_key, aws_region):
    """Use Claude v2 LLM to extract technical skills."""
    return run_skill_extraction_prompt(text, aws_access_key, aws_secret_key, aws_region)


def generate_match_report(resume_texts, job_text, aws_access_key, aws_secret_key, aws_region):
    results = []
    jd_skills = run_skill_extraction_prompt(job_text, aws_access_key, aws_secret_key, aws_region)

    for resume_name, resume_text in resume_texts.items():
        if not resume_text.strip():
            print(f"⚠️ Empty text extracted from resume: {resume_name}, skipping.")
            continue
        if not job_text.strip():
            print(f"⚠️ Empty job description text, skipping matching.")
            continue

        resume_skills = run_skill_extraction_prompt(resume_text, aws_access_key, aws_secret_key, aws_region)
        matched_skills = list(set(jd_skills).intersection(set(resume_skills)))
        missing_skills = list(set(jd_skills) - set(resume_skills))

        embedding_score = round(
            calculate_similarity(resume_text, job_text, aws_access_key, aws_secret_key, aws_region) * 100, 2
        )
        match_score = round(len(matched_skills) / len(jd_skills) * 100, 2) if jd_skills else 0.0

        results.append({
            "resume": resume_name,
            "embedding_score": embedding_score,
            "match_score": match_score,
            "all_resume_skills": resume_skills,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills
        })

    return results
