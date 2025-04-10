
# 📄 AI Profile Syncer - README

## 📈 Resume Matching Scoring Explained

This application uses a hybrid scoring approach to evaluate how well a resume matches a given job description. It combines:

- **Match Score based on overlapping technical skills**
- **Cosine Similarity based on semantic embedding vectors**

---

### 🔹 1. Match Score (%)

This score is calculated based on the intersection of skills extracted from the resume and those from the job description.

**Formula:**

```
Match Score (%) = (|Matching Skills| ÷ |JD Skills|) × 100
```

**Example:**

- JD Skills: Python, AWS, Docker, SQL, Git  
- Resume Skills: Python, AWS, SQL, Flask

- Matching Skills: Python, AWS, SQL → count = 3  
- Total JD Skills: 5

```
Match Score = (3 ÷ 5) × 100 = 60%
```

**Notes:**

- This gives a direct percentage of how many JD-required skills are found in the resume.
- Extracted skills are obtained using Claude v2 from AWS Bedrock.

---

### 🔹 2. Cosine Similarity (%)

This is a semantic similarity score that captures the meaning-level match between the job description and the resume.

**Steps:**

1. Use AWS Bedrock Titan Embeddings G1 - Text to convert the job description and resume into high-dimensional vectors.
2. Cosine similarity is computed between the two embedding vectors.

**Formula:**

```
Cosine Similarity = (A · B) / (‖A‖ × ‖B‖)
```

Where:

- A, B = embedding vectors for resume and job description
- · = dot product
- ‖A‖ = magnitude (Euclidean norm) of vector A

**Final score is scaled to a percentage:**

```
Cosine Similarity (%) = Cosine Similarity × 100
```

---

### 🔹 How These Scores Are Used

Both scores are shown in the UI:

- **Match Score (%)** → based on skill overlap
- **Cosine Similarity (%)** → based on full-text meaning

**Filtering:**

- The user can set a minimum Match Score threshold via the UI.
- Only resumes above that threshold are saved to `selected_profile/` directory.

---

### 📁 Example Report Output

| Resume       | Match Score (%) | Cosine Similarity (%) | Matching Skills    | Missing Skills        |
|--------------|------------------|------------------------|---------------------|------------------------|
| resume1.pdf  | 65               | 71.29                  | Python, SQL         | Docker, Git, AWS      |
| resume2.docx | 85               | 78.92                  | Python, SQL, AWS    | Docker, Git           |
