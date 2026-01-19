"""
AIHawk Resume Builder - Web Application
Lightweight FastAPI backend for generating AI-powered resumes
"""
import os
from pathlib import Path
from typing import Optional
from string import Template

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="AIHawk Resume Builder",
    description="AI-powered resume and cover letter generator",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


# Pydantic models
class GenerateResumeRequest(BaseModel):
    resume_yaml: str
    style: str = "Classic"
    api_key: Optional[str] = None
    job_description: Optional[str] = None


# HTML Template
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resume</title>
    <style>
        $style_css
    </style>
</head>
$body
</html>"""

# Simple CSS styles
STYLES = {
    "Classic": """
        body { font-family: 'Georgia', serif; max-width: 800px; margin: 0 auto; padding: 40px; line-height: 1.6; }
        header { border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 20px; }
        h1 { margin: 0; color: #333; } h2 { color: #555; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
        .contact { color: #666; } section { margin-bottom: 25px; }
        ul { padding-left: 20px; } li { margin-bottom: 8px; }
    """,
    "Modern": """
        body { font-family: 'Helvetica Neue', sans-serif; max-width: 800px; margin: 0 auto; padding: 40px; line-height: 1.5; color: #333; }
        header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; margin: -40px -40px 30px; }
        h1 { margin: 0; font-weight: 300; font-size: 2.5em; } h2 { color: #667eea; font-weight: 500; }
        .contact { opacity: 0.9; } section { margin-bottom: 25px; }
        ul { padding-left: 20px; } li { margin-bottom: 8px; }
    """
}


def get_api_key(request_key: Optional[str] = None) -> str:
    api_key = request_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="API key required")
    return api_key


def parse_resume_yaml(yaml_str: str) -> dict:
    try:
        return yaml.safe_load(yaml_str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {str(e)}")


def generate_resume_with_openai(resume_data: dict, api_key: str, job_description: str = None) -> str:
    """Generate HTML resume using OpenAI API directly."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    prompt = f"""You are an expert resume writer. Generate a professional HTML resume body (just the <body> content, no <html> or <head> tags) based on this data:

Resume Data:
{yaml.dump(resume_data, default_flow_style=False)}

{"Job Description to tailor for:" + job_description if job_description else ""}

Generate clean, semantic HTML with:
- A <header> section with name, contact info
- <section> elements for Education, Experience, Skills, Projects (if data exists)
- Use <h1> for name, <h2> for section titles
- Use <ul>/<li> for lists
- Add class="contact" to contact info

Return ONLY the HTML body content, no explanations."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=4000
    )

    return response.choices[0].message.content


def generate_cover_letter_with_openai(resume_data: dict, job_description: str, api_key: str) -> str:
    """Generate HTML cover letter using OpenAI API."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    name = resume_data.get('personal_information', {}).get('name', 'Applicant')
    surname = resume_data.get('personal_information', {}).get('surname', '')

    prompt = f"""You are an expert cover letter writer. Generate a professional HTML cover letter body based on:

Applicant: {name} {surname}
Resume Data:
{yaml.dump(resume_data, default_flow_style=False)}

Job Description:
{job_description}

Generate a compelling cover letter in HTML format with:
- Professional greeting
- 3-4 paragraphs highlighting relevant experience
- Strong closing
- Use <p> tags for paragraphs

Return ONLY the HTML body content."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=2000
    )

    return response.choices[0].message.content


# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = Path(__file__).parent / "static" / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse(content="<h1>AIHawk Resume Builder</h1><p>Visit <a href='/docs'>/docs</a></p>")


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "AIHawk Resume Builder"}


@app.get("/api/styles")
async def get_styles():
    return {"styles": [{"name": name, "author": "AIHawk"} for name in STYLES.keys()]}


@app.post("/api/generate/resume")
async def generate_resume(request: GenerateResumeRequest):
    try:
        api_key = get_api_key(request.api_key)
        resume_data = parse_resume_yaml(request.resume_yaml)
        style_css = STYLES.get(request.style, STYLES["Classic"])

        body_html = generate_resume_with_openai(resume_data, api_key)

        template = Template(HTML_TEMPLATE)
        full_html = template.substitute(body=body_html, style_css=style_css)

        return {"success": True, "html": full_html}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate/resume-tailored")
async def generate_tailored_resume(request: GenerateResumeRequest):
    try:
        if not request.job_description:
            raise HTTPException(status_code=400, detail="Job description required")

        api_key = get_api_key(request.api_key)
        resume_data = parse_resume_yaml(request.resume_yaml)
        style_css = STYLES.get(request.style, STYLES["Classic"])

        body_html = generate_resume_with_openai(resume_data, api_key, request.job_description)

        template = Template(HTML_TEMPLATE)
        full_html = template.substitute(body=body_html, style_css=style_css)

        return {"success": True, "html": full_html}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate/cover-letter")
async def generate_cover_letter(request: GenerateResumeRequest):
    try:
        if not request.job_description:
            raise HTTPException(status_code=400, detail="Job description required")

        api_key = get_api_key(request.api_key)
        resume_data = parse_resume_yaml(request.resume_yaml)
        style_css = STYLES.get(request.style, STYLES["Classic"])

        body_html = generate_cover_letter_with_openai(resume_data, request.job_description, api_key)

        template = Template(HTML_TEMPLATE)
        full_html = template.substitute(body=f"<body>{body_html}</body>", style_css=style_css)

        return {"success": True, "html": full_html}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sample-resume")
async def get_sample_resume():
    return {"sample_yaml": """personal_information:
  name: "John"
  surname: "Doe"
  email: "john.doe@email.com"
  phone: "+1 555-123-4567"
  city: "San Francisco"
  country: "USA"
  linkedin: "linkedin.com/in/johndoe"
  github: "github.com/johndoe"

education_details:
  - education_level: "Bachelor's Degree"
    institution: "University of California"
    field_of_study: "Computer Science"
    final_evaluation_grade: "3.8 GPA"
    year_of_completion: 2020

experience_details:
  - position: "Senior Software Engineer"
    company: "Tech Corp"
    employment_period: "2020 - Present"
    location: "San Francisco, CA"
    key_responsibilities:
      - responsibility: "Led development of microservices architecture"
      - responsibility: "Mentored junior developers"
    skills_acquired:
      - "Python"
      - "FastAPI"
      - "AWS"

projects:
  - name: "AI Resume Builder"
    description: "Open-source tool for generating professional resumes"
    link: "github.com/johndoe/resume-builder"

languages:
  - language: "English"
    proficiency: "Native"
  - language: "Spanish"
    proficiency: "Intermediate"
"""}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
