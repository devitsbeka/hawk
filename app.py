"""
AIHawk Resume Builder - Web Application
FastAPI backend for generating AI-powered resumes and cover letters
"""
import os
import base64
import hashlib
import tempfile
from pathlib import Path
from typing import Optional
from string import Template

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import local modules
from src.resume_schemas.resume import Resume
from src.libs.resume_and_cover_builder import ResumeGenerator, StyleManager
from src.libs.resume_and_cover_builder.config import global_config
from src.libs.resume_and_cover_builder.llm.llm_generate_resume import LLMResumer
from src.libs.resume_and_cover_builder.llm.llm_generate_resume_from_job import LLMResumeJobDescription
from src.libs.resume_and_cover_builder.llm.llm_generate_cover_letter_from_job import LLMCoverLetterJobDescription
from src.libs.resume_and_cover_builder.module_loader import load_module

app = FastAPI(
    title="AIHawk Resume Builder",
    description="AI-powered resume and cover letter generator",
    version="1.0.0"
)

# Add CORS middleware
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


# Initialize global config paths
def init_global_config(api_key: str):
    """Initialize the global configuration with paths and API key."""
    lib_directory = Path(__file__).parent / "src" / "libs" / "resume_and_cover_builder"
    global_config.STRINGS_MODULE_RESUME_PATH = lib_directory / "resume_prompt/strings_feder-cr.py"
    global_config.STRINGS_MODULE_RESUME_JOB_DESCRIPTION_PATH = lib_directory / "resume_job_description_prompt/strings_feder-cr.py"
    global_config.STRINGS_MODULE_COVER_LETTER_JOB_DESCRIPTION_PATH = lib_directory / "cover_letter_prompt/strings_feder-cr.py"
    global_config.STRINGS_MODULE_NAME = "strings_feder_cr"
    global_config.STYLES_DIRECTORY = lib_directory / "resume_style"
    global_config.LOG_OUTPUT_FILE_PATH = Path(tempfile.gettempdir())
    global_config.API_KEY = api_key


# Pydantic models for API
class ResumeData(BaseModel):
    """Resume data model for API requests."""
    personal_information: dict
    education_details: Optional[list] = None
    experience_details: Optional[list] = None
    projects: Optional[list] = None
    achievements: Optional[list] = None
    certifications: Optional[list] = None
    languages: Optional[list] = None
    interests: Optional[list] = None


class GenerateResumeRequest(BaseModel):
    """Request model for generating a resume."""
    resume_yaml: str
    style: str = "Classic"
    api_key: Optional[str] = None


class GenerateTailoredResumeRequest(BaseModel):
    """Request model for generating a job-tailored resume."""
    resume_yaml: str
    job_description: str
    style: str = "Classic"
    api_key: Optional[str] = None


class GenerateCoverLetterRequest(BaseModel):
    """Request model for generating a cover letter."""
    resume_yaml: str
    job_description: str
    style: str = "Classic"
    api_key: Optional[str] = None


def get_api_key(request_key: Optional[str] = None) -> str:
    """Get API key from request or environment."""
    api_key = request_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="API key required. Provide it in the request or set OPENAI_API_KEY environment variable."
        )
    return api_key


def get_style_path(style_name: str) -> Path:
    """Get the path to a style CSS file."""
    style_manager = StyleManager()
    styles = style_manager.get_styles()

    # Find matching style (case-insensitive partial match)
    for name, (file_name, _) in styles.items():
        if style_name.lower() in name.lower():
            return style_manager.styles_directory / file_name

    # Default to first available style
    if styles:
        first_style = list(styles.values())[0]
        return style_manager.styles_directory / first_style[0]

    raise HTTPException(status_code=404, detail=f"Style '{style_name}' not found")


def generate_html_from_resume(resume_yaml: str, api_key: str, style_path: Path, job_description: str = None, is_cover_letter: bool = False) -> str:
    """Generate HTML from resume YAML using LLM."""
    init_global_config(api_key)

    # Parse resume
    try:
        resume_object = Resume(resume_yaml)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid resume YAML: {str(e)}")

    # Read style CSS
    try:
        with open(style_path, "r") as f:
            style_css = f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Style file not found: {style_path}")

    # Generate HTML based on type
    if is_cover_letter and job_description:
        # Generate cover letter
        strings = load_module(global_config.STRINGS_MODULE_COVER_LETTER_JOB_DESCRIPTION_PATH, global_config.STRINGS_MODULE_NAME)
        gpt_answerer = LLMCoverLetterJobDescription(api_key, strings)
        gpt_answerer.set_resume(resume_object)
        gpt_answerer.set_job_description_from_text(job_description)
        body_html = gpt_answerer.generate_cover_letter()
    elif job_description:
        # Generate job-tailored resume
        strings = load_module(global_config.STRINGS_MODULE_RESUME_JOB_DESCRIPTION_PATH, global_config.STRINGS_MODULE_NAME)
        gpt_answerer = LLMResumeJobDescription(api_key, strings)
        gpt_answerer.set_resume(resume_object)
        gpt_answerer.set_job_description_from_text(job_description)
        body_html = gpt_answerer.generate_html_resume()
    else:
        # Generate base resume
        strings = load_module(global_config.STRINGS_MODULE_RESUME_PATH, global_config.STRINGS_MODULE_NAME)
        gpt_answerer = LLMResumer(api_key, strings)
        gpt_answerer.set_resume(resume_object)
        body_html = gpt_answerer.generate_html_resume()

    # Apply template
    template = Template(global_config.html_template)
    full_html = template.substitute(body=body_html, style_css=style_css)

    return full_html


# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page."""
    index_path = Path(__file__).parent / "static" / "index.html"
    if index_path.exists():
        with open(index_path, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="""
    <html>
        <head><title>AIHawk Resume Builder</title></head>
        <body>
            <h1>AIHawk Resume Builder</h1>
            <p>API is running. Visit <a href="/docs">/docs</a> for API documentation.</p>
        </body>
    </html>
    """)


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "AIHawk Resume Builder"}


@app.get("/api/styles")
async def get_styles():
    """Get available resume styles."""
    style_manager = StyleManager()
    styles = style_manager.get_styles()
    return {
        "styles": [
            {"name": name, "author": author_link}
            for name, (_, author_link) in styles.items()
        ]
    }


@app.post("/api/generate/resume")
async def generate_resume(request: GenerateResumeRequest):
    """Generate a base resume from YAML data."""
    try:
        api_key = get_api_key(request.api_key)
        style_path = get_style_path(request.style)

        html = generate_html_from_resume(
            resume_yaml=request.resume_yaml,
            api_key=api_key,
            style_path=style_path
        )

        return {
            "success": True,
            "html": html,
            "message": "Resume generated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate/resume-tailored")
async def generate_tailored_resume(request: GenerateTailoredResumeRequest):
    """Generate a job-tailored resume from YAML data and job description."""
    try:
        api_key = get_api_key(request.api_key)
        style_path = get_style_path(request.style)

        html = generate_html_from_resume(
            resume_yaml=request.resume_yaml,
            api_key=api_key,
            style_path=style_path,
            job_description=request.job_description
        )

        return {
            "success": True,
            "html": html,
            "message": "Tailored resume generated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate/cover-letter")
async def generate_cover_letter(request: GenerateCoverLetterRequest):
    """Generate a cover letter from YAML data and job description."""
    try:
        api_key = get_api_key(request.api_key)
        style_path = get_style_path(request.style)

        html = generate_html_from_resume(
            resume_yaml=request.resume_yaml,
            api_key=api_key,
            style_path=style_path,
            job_description=request.job_description,
            is_cover_letter=True
        )

        return {
            "success": True,
            "html": html,
            "message": "Cover letter generated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sample-resume")
async def get_sample_resume():
    """Get a sample resume YAML for testing."""
    sample_path = Path(__file__).parent / "data_folder_example" / "plain_text_resume.yaml"
    if sample_path.exists():
        with open(sample_path, "r") as f:
            return {"sample_yaml": f.read()}
    return {"sample_yaml": """personal_information:
  name: "John"
  surname: "Doe"
  email: "john.doe@email.com"
  phone: "+1 555-123-4567"
  city: "San Francisco"
  country: "USA"
  linkedin: "https://linkedin.com/in/johndoe"
  github: "https://github.com/johndoe"

education_details:
  - education_level: "Bachelor's Degree"
    institution: "University of California"
    field_of_study: "Computer Science"
    year_of_completion: 2020

experience_details:
  - position: "Software Engineer"
    company: "Tech Corp"
    employment_period: "2020 - Present"
    location: "San Francisco, CA"
    industry: "Technology"
    key_responsibilities:
      - responsibility: "Developed web applications using React and Node.js"
      - responsibility: "Collaborated with cross-functional teams"
    skills_acquired:
      - "React"
      - "Node.js"
      - "TypeScript"

projects:
  - name: "Open Source Project"
    description: "A popular open-source library for data processing"
    link: "https://github.com/johndoe/project"

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
