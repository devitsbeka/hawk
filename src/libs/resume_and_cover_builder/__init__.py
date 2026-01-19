__version__ = '0.1'

# Import all the necessary classes and functions, called when the package is imported
from .resume_generator import ResumeGenerator
from .style_manager import StyleManager

# ResumeFacade requires selenium, make it optional
try:
    from .resume_facade import ResumeFacade
except ImportError:
    ResumeFacade = None