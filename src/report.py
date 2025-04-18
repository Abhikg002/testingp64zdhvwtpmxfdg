from src.bedrock_llm import run_skill_extraction_prompt
from src.matching import calculate_similarity
from src.logger import logger
import time
from src.matching import calculate_similarity

def extract_skills(text, aws_access_key, aws_secret_key, aws_region):
    """Use Claude v2 LLM to extract technical skills."""
    return run_skill_extraction_prompt(text, aws_access_key, aws_secret_key, aws_region)


def generate_match_report(resume_texts, job_text, aws_access_key, aws_secret_key, aws_region, selected_skills):
    results = []
    # Extract all JD skills for basic match score
    all_jd_skills = set(extract_skills(job_text, aws_access_key, aws_secret_key, aws_region))
    
    for resume_name, resume_text in resume_texts.items():
        if not resume_text.strip() or not job_text.strip():
            continue

        resume_skills = run_skill_extraction_prompt(resume_text, aws_access_key, aws_secret_key, aws_region)
        
        # Calculate basic match score using ALL extracted JD skills
        matched_skills_all = list(set(all_jd_skills).intersection(set(resume_skills)))
        missing_skills = list(set(all_jd_skills) - set(resume_skills))
        match_score = round(len(matched_skills_all) / len(all_jd_skills) * 100, 2) if all_jd_skills else 0.0

        # Calculate weighted score using only selected skills and their weights
        if selected_skills:
            matched_selected_skills = list(set(selected_skills.keys()).intersection(set(resume_skills)))
            total_weight = sum(selected_skills.values())  # Sum of all selected skill weights
            weighted_score = sum(selected_skills[skill] for skill in matched_selected_skills)
            weighted_percentage = round((weighted_score / total_weight) * 100, 2) if total_weight > 0 else 0
        else:
            weighted_percentage = 0

        embedding_score = round(
            calculate_similarity(resume_text, job_text, aws_access_key, aws_secret_key, aws_region) * 100, 2
        )

        results.append({
            "resume": resume_name,
            "embedding_score": embedding_score,
            "match_score": match_score,
            "weighted_score": weighted_percentage,
            "all_resume_skills": resume_skills,
            "matched_skills": matched_skills_all,
            "missing_skills": missing_skills
        })

    return results