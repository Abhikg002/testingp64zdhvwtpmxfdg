import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.embeddings import generate_embeddings

def calculate_similarity(resume_text, job_text, aws_access_key, aws_secret_key, aws_region):
    if not resume_text.strip() or not job_text.strip():
        return 0.0  # Avoid API call and prevent Bedrock error

    job_embedding = np.array(generate_embeddings(job_text, aws_access_key, aws_secret_key, aws_region)).reshape(1, -1)
    resume_embedding = np.array(generate_embeddings(resume_text, aws_access_key, aws_secret_key, aws_region)).reshape(1, -1)

    similarity = np.dot(resume_embedding, job_embedding.T) / (
        np.linalg.norm(resume_embedding) * np.linalg.norm(job_embedding)
    )
    return float(similarity[0][0])
