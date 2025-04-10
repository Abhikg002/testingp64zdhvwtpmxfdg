### Features
```commandline
✅ Users can upload a ZIP file containing multiple job descriptions and resumes.
✅ The system extracts JDs/resumes from ZIP and processes them automatically.
✅ Works alongside individual JDs/resume uploads (both options available).

✅ Automatically extracts required skills from the job description

✅ Extracts skills from resumes and compares them
✅ Calculates & displays the Match Skills(%) and Resume Similarity(%)
✅ Shows Matched & Missing skills per resume
✅ Recruiters can easily see gaps & strengths
✅ Allows downloading a CSV report & Selected Resumes based on threshold value
```

#### Repository
~~~
resume-matching/
│── ui.py                # Main script to run resume matching UI
│── requirements.txt      # Dependencies  
│── .env                 # Stores AWS credentials  
│── data/                # Stores resumes & job descriptions  
│   │── resumes/  
│   │── job_descriptions/  
│── src/                 # Source code  
│   │── bedrock_llm.py   # LLM logic  
│   │── parser.py        # Resume parsing logic  
│   │── embeddings.py    # AWS Bedrock embedding generation  
│   │── matching.py      # Similarity matching logic  
│   │── report.py        # Report generation logic
│   │── utils.py        # to clear folder
│── config.py            # Load environment variables  
│── README.md            # Project documentation  
~~~

#### Install
``
pip install -r requirements.txt
``

#### Run
```
streamlit run ui.py
```