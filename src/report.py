from src.bedrock_llm import run_skill_extraction_prompt
from src.matching import calculate_similarity
from src.logger import logger
import time

def extract_skills(text):
    """Use Claude v2 LLM to extract technical skills."""
    return run_skill_extraction_prompt(text)

def generate_match_report(resume_texts, job_text):
    """Generate match report with skill comparison."""
    results = []

    logger.info(f"Processing job_text: {job_text}")
    job_skills = extract_skills(job_text)
    logger.info(f"Extracted job description skills: {job_skills}")
    job_skills_lower = {skill.lower() for skill in job_skills}

    for file_name, resume_text in resume_texts.items():
        logger.info(f"Processing resume name: {file_name}")
        resume_skills = extract_skills(resume_text)
        time.sleep(1.5)
        logger.info(f"Extracted resume skills: {resume_skills}")
        resume_skills_lower = {skill.lower() for skill in resume_skills}

        matched_skills = resume_skills_lower.intersection(job_skills_lower)
        missing_skills = job_skills_lower - resume_skills_lower

        skill_match_score = round((len(matched_skills) / len(job_skills_lower)) * 100, 2) if job_skills_lower else 0
        embedding_score = round(calculate_similarity(resume_text, job_text) * 100, 2)
        time.sleep(1.5)

        results.append({
            "resume": file_name,
            "match_score": skill_match_score,  # Main score based on skill overlap
            "embedding_score": embedding_score,  # Optional: cosine similarity
            "all_resume_skills": list(resume_skills_lower),
            "matched_skills": list(matched_skills),
            "missing_skills": list(missing_skills)
        })

    results.sort(key=lambda x: x["match_score"], reverse=True)
    return results
