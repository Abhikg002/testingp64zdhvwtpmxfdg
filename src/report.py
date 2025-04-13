from src.bedrock_llm import run_skill_extraction_prompt
from src.matching import calculate_similarity
from src.logger import logger
import time

def extract_skills(text, aws_access_key, aws_secret_key, aws_region):
    """Use Claude v2 LLM to extract technical skills."""
    return run_skill_extraction_prompt(text, aws_access_key, aws_secret_key, aws_region)

def generate_match_report(resume_texts, job_text, aws_access_key, aws_secret_key, aws_region, selected_skills=None):
    """
    Generate match report with skill comparison.
    
    Parameters:
        resume_texts (dict): Dictionary of resume texts
        job_text (str): Job description text
        aws_access_key (str): AWS access key
        aws_secret_key (str): AWS secret key
        aws_region (str): AWS region
        selected_skills (dict, optional): Dictionary of selected skills with weights {skill: weight}
    """
    results = []

    logger.info("Extracting skills from job description")
    # If selected_skills is provided, we'll use those instead of extracting again
    if selected_skills:
        job_skills_with_weights = selected_skills
        job_skills_lower = {skill.lower(): weight for skill, weight in job_skills_with_weights.items()}
    else:
        job_skills = extract_skills(job_text, aws_access_key, aws_secret_key, aws_region)
        job_skills_lower = {skill.lower(): 1.0 for skill in job_skills}  # Default weight is 1.0

    for file_name, resume_text in resume_texts.items():
        logger.info(f"Processing resume: {file_name}")
        resume_skills = extract_skills(resume_text, aws_access_key, aws_secret_key, aws_region)
        time.sleep(1.5)

        resume_skills_lower = {skill.lower() for skill in resume_skills}

        # Calculate weighted skill match
        matched_skills = resume_skills_lower.intersection(job_skills_lower.keys())
        missing_skills = set(job_skills_lower.keys()) - resume_skills_lower
        
        # Apply weights to matched skills
        total_weight = sum(job_skills_lower.values())
        if total_weight > 0:
            weighted_match = sum(job_skills_lower[skill] for skill in matched_skills)
            skill_match_score = round((weighted_match / total_weight) * 100, 2)
        else:
            skill_match_score = 0
            
        # Store the weights that were applied to this match
        skill_weights = {skill: job_skills_lower[skill] for skill in matched_skills} if matched_skills else {}
        
        embedding_score = round(calculate_similarity(resume_text, job_text, aws_access_key, aws_secret_key, aws_region) * 100, 2)
        time.sleep(1.5)

        results.append({
            "resume": file_name,
            "match_score": skill_match_score,
            "embedding_score": embedding_score,
            "all_resume_skills": list(resume_skills_lower),
            "matched_skills": list(matched_skills),
            "missing_skills": list(missing_skills),
            "skill_weights": str(skill_weights) if skill_weights else "No matches"
        })

    results.sort(key=lambda x: x["match_score"], reverse=True)
    return results