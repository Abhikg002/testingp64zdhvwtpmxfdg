import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.embeddings import generate_embeddings

def calculate_similarity(resume_text, job_text, aws_access_key, aws_secret_key, aws_region):
    """Compute cosine similarity between resume and job description embeddings."""
    resume_embedding = np.array(generate_embeddings(resume_text, aws_access_key, aws_secret_key, aws_region)).reshape(1, -1)
    job_embedding = np.array(generate_embeddings(job_text, aws_access_key, aws_secret_key, aws_region)).reshape(1, -1)
    
    similarity_score = cosine_similarity(resume_embedding, job_embedding)[0][0]

    return similarity_score