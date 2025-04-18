from src.bedrock_llm import run_skill_extraction_prompt
from src.matching import calculate_similarity
from src.logger import logger
import time
from src.matching import calculate_similarity
from src.bedrock_llm import feedback_generation

def extract_skills(text, aws_access_key, aws_secret_key, aws_region):
    """Use Claude v2 LLM to extract technical skills."""
    return run_skill_extraction_prompt(text, aws_access_key, aws_secret_key, aws_region)


def generate_match_report(resume_texts, job_text, aws_access_key, aws_secret_key, aws_region):
    results = []
    logger.info("Extracting skills from job description")
    jd_skills = run_skill_extraction_prompt(job_text, aws_access_key, aws_secret_key, aws_region)
    jd_skills_lower = {skill.lower() for skill in jd_skills["Technical Skills"]}

    for resume_name, resume_text in resume_texts.items():
        if not resume_text.strip():
            print(f"⚠️ Empty text extracted from resume: {resume_name}, skipping.")
            continue
        if not job_text.strip():
            print(f"⚠️ Empty job description text, skipping matching.")
            continue

        resume_skills = run_skill_extraction_prompt(resume_text, aws_access_key, aws_secret_key, aws_region)
        resume_skills_lower = {skill.lower() for skill in resume_skills["Technical Skills"]}

        matched_skills = list(set(jd_skills_lower).intersection(set(resume_skills_lower)))
        missing_skills = list(set(jd_skills_lower) - set(resume_skills_lower))

        embedding_score = round(
            calculate_similarity(resume_text, job_text, aws_access_key, aws_secret_key, aws_region) * 100, 2
        )
        match_score = round((len(matched_skills) / len(jd_skills_lower)) * 100, 2) if jd_skills_lower else 0.0

        feedback = feedback_generation(resume_text, job_text, aws_access_key, aws_secret_key, aws_region)

        results.append({
            "resume": resume_name,
            "name" : resume_skills["Name"],
            "email": resume_skills["Email"],
            "location":resume_skills["Location"],
            "years of experience": resume_skills["Years of Experience"],
            "embedding_score": embedding_score,
            "match_score": match_score,
            "all_resume_skills": resume_skills_lower,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "feedback":feedback
        })

    return results
