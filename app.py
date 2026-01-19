"""
AIHawk Resume Builder - Web API
Uses the existing codebase for AI-powered resume generation
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

# Import from existing codebase
from src.resume_schemas.resume import Resume
from src.libs.resume_and_cover_builder.llm.llm_generate_resume import LLMResumer
from src.libs.resume_and_cover_builder.module_loader import load_module
from src.libs.resume_and_cover_builder.config import global_config

app = FastAPI(title="AIHawk Resume Builder", version="1.0.0")

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


def init_config():
    """Initialize global config paths."""
    lib_dir = Path(__file__).parent / "src" / "libs" / "resume_and_cover_builder"
    global_config.STRINGS_MODULE_RESUME_PATH = lib_dir / "resume_prompt/strings_feder-cr.py"
    global_config.STRINGS_MODULE_NAME = "strings_feder_cr"
    global_config.STYLES_DIRECTORY = lib_dir / "resume_style"


# Request models
class GenerateRequest(BaseModel):
    resume_yaml: str
    api_key: Optional[str] = None
    job_description: Optional[str] = None


# CSS Styles
STYLE_CSS = """
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 850px; margin: 0 auto; padding: 40px; line-height: 1.6; color: #333; }
header { border-bottom: 3px solid #2c3e50; padding-bottom: 20px; margin-bottom: 30px; }
h1 { margin: 0 0 10px 0; color: #2c3e50; font-size: 2.2em; }
h2 { color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 8px; margin-top: 25px; font-size: 1.3em; }
.contact { color: #7f8c8d; font-size: 0.95em; }
.contact a { color: #3498db; text-decoration: none; }
section { margin-bottom: 25px; }
ul { padding-left: 20px; margin: 10px 0; }
li { margin-bottom: 8px; }
.job-title { font-weight: 600; color: #2c3e50; }
.company { color: #7f8c8d; }
.date { color: #95a5a6; font-size: 0.9em; }
@media print { body { padding: 20px; } }
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resume</title>
    <style>$style_css</style>
</head>
$body
</html>"""


def get_api_key(request_key: Optional[str] = None) -> str:
    api_key = request_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key required")
    return api_key


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = Path(__file__).parent / "static" / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse("<h1>AIHawk Resume Builder</h1><p>API running. See <a href='/docs'>/docs</a></p>")


@app.get("/api/health")
async def health():
    return {"status": "healthy"}


@app.get("/api/styles")
async def styles():
    return {"styles": [{"name": "Professional", "author": "AIHawk"}]}


@app.post("/api/generate/resume")
async def generate_resume(request: GenerateRequest):
    try:
        init_config()
        api_key = get_api_key(request.api_key)

        # Parse resume YAML
        resume_object = Resume(request.resume_yaml)

        # Load prompt strings
        strings = load_module(global_config.STRINGS_MODULE_RESUME_PATH, global_config.STRINGS_MODULE_NAME)

        # Generate with LLM
        llm_resumer = LLMResumer(api_key, strings)
        llm_resumer.set_resume(resume_object)
        body_html = llm_resumer.generate_html_resume()

        # Apply template
        template = Template(HTML_TEMPLATE)
        full_html = template.substitute(body=body_html, style_css=STYLE_CSS)

        return {"success": True, "html": full_html}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate/resume-tailored")
async def generate_tailored(request: GenerateRequest):
    try:
        if not request.job_description:
            raise HTTPException(status_code=400, detail="Job description required")

        init_config()
        api_key = get_api_key(request.api_key)

        # Parse resume
        resume_object = Resume(request.resume_yaml)

        # Load job-tailored prompt strings
        lib_dir = Path(__file__).parent / "src" / "libs" / "resume_and_cover_builder"
        strings = load_module(lib_dir / "resume_job_description_prompt/strings_feder-cr.py", global_config.STRINGS_MODULE_NAME)

        # Import and use job-tailored generator
        from src.libs.resume_and_cover_builder.llm.llm_generate_resume_from_job import LLMResumeJobDescription

        llm_resumer = LLMResumeJobDescription(api_key, strings)
        llm_resumer.set_resume(resume_object)
        llm_resumer.set_job_description_from_text(request.job_description)
        body_html = llm_resumer.generate_html_resume()

        template = Template(HTML_TEMPLATE)
        full_html = template.substitute(body=body_html, style_css=STYLE_CSS)

        return {"success": True, "html": full_html}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate/cover-letter")
async def generate_cover_letter(request: GenerateRequest):
    try:
        if not request.job_description:
            raise HTTPException(status_code=400, detail="Job description required")

        init_config()
        api_key = get_api_key(request.api_key)

        resume_object = Resume(request.resume_yaml)

        lib_dir = Path(__file__).parent / "src" / "libs" / "resume_and_cover_builder"
        strings = load_module(lib_dir / "cover_letter_prompt/strings_feder-cr.py", global_config.STRINGS_MODULE_NAME)

        from src.libs.resume_and_cover_builder.llm.llm_generate_cover_letter_from_job import LLMCoverLetterJobDescription

        llm_generator = LLMCoverLetterJobDescription(api_key, strings)
        llm_generator.set_resume(resume_object)
        llm_generator.set_job_description_from_text(request.job_description)
        body_html = llm_generator.generate_cover_letter()

        template = Template(HTML_TEMPLATE)
        full_html = template.substitute(body=f"<body>{body_html}</body>", style_css=STYLE_CSS)

        return {"success": True, "html": full_html}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sample-resume")
async def sample_resume():
    sample_path = Path(__file__).parent / "data_folder_example" / "plain_text_resume.yaml"
    if sample_path.exists():
        return {"sample_yaml": sample_path.read_text()}
    return {"sample_yaml": "personal_information:\n  name: John\n  surname: Doe\n  email: john@example.com"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
