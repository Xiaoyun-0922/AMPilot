"""
Code Execution Tool for Data Analysis Agent

Safe Python code execution tool supporting data analysis and statistical computing
"""

import os
import sys
import subprocess
import tempfile
import traceback
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO, BytesIO
import base64

import sys
sys.path.append('..')
from configuration import CODE_EXECUTION_TIMEOUT, TEMP_CODE_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)

class CodeExecutor:
    """Safe Python code executor"""

    def __init__(self):
        """Initialize code executor"""
        self.temp_dir = Path(TEMP_CODE_DIR)
        self.results_dir = Path(RESULTS_DIR)
        self.temp_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)

        # Pre-imported safe modules
        self.safe_imports = {
            'pandas': pd,
            'numpy': np,
            'matplotlib.pyplot': plt,
            'seaborn': sns,
            'scipy.stats': None,  # Lazy import
            'sklearn': None,      # Lazy import
            'statsmodels': None,  # Lazy import
        }

        logger.info("Code executor initialized successfully")
    
    def execute_code(self, code: str, data_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Safely execute Python code

        Args:
            code: Python code to execute
            data_context: Data context containing dataframes etc.

        Returns:
            Execution result dictionary
        """
        try:
            # Create temporary file
            temp_file = self.temp_dir / f"analysis_{os.getpid()}.py"

            # Prepare execution environment
            execution_globals = self._prepare_execution_environment(data_context)

            # Capture output
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            stdout_capture = StringIO()
            stderr_capture = StringIO()

            try:
                sys.stdout = stdout_capture
                sys.stderr = stderr_capture

                # Execute code
                exec(code, execution_globals)

                # Get output
                stdout_output = stdout_capture.getvalue()
                stderr_output = stderr_capture.getvalue()

                # Save plots
                plots = self._save_plots()

                return {
                    'success': True,
                    'stdout': stdout_output,
                    'stderr': stderr_output,
                    'plots': plots,
                    'variables': self._extract_variables(execution_globals),
                    'error': None
                }

            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        except Exception as e:
            logger.error(f"Code execution error: {e}")
            return {
                'success': False,
                'stdout': '',
                'stderr': str(e),
                'plots': [],
                'variables': {},
                'error': traceback.format_exc()
            }
    
    def _prepare_execution_environment(self, data_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare code execution environment"""
        # Basic environment
        env = {
            '__builtins__': __builtins__,
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
        }

        # Lazy import scientific computing libraries
        try:
            import scipy.stats as stats
            env['stats'] = stats
        except ImportError:
            pass

        try:
            import sklearn
            env['sklearn'] = sklearn
        except ImportError:
            pass

        try:
            import statsmodels.api as sm
            env['sm'] = sm
        except ImportError:
            pass

        # Add data context
        if data_context:
            env.update(data_context)

        return env
    
    def _save_plots(self) -> list:
        """Save matplotlib plots"""
        plots = []

        # Get all figures
        for i, fig in enumerate(plt.get_fignums()):
            try:
                figure = plt.figure(fig)

                # Save as base64 encoded image
                buffer = BytesIO()
                figure.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)

                # Convert to base64
                plot_data = base64.b64encode(buffer.getvalue()).decode()
                plots.append({
                    'figure_id': fig,
                    'data': plot_data,
                    'format': 'png'
                })

                buffer.close()

            except Exception as e:
                logger.warning(f"Failed to save plot {fig}: {e}")

        # Clear all figures
        plt.close('all')

        return plots
    
    def _extract_variables(self, execution_globals: Dict[str, Any]) -> Dict[str, Any]:
        """Extract important variables after execution"""
        variables = {}

        for name, value in execution_globals.items():
            # Skip built-in variables and modules
            if name.startswith('_') or hasattr(value, '__module__'):
                continue

            try:
                # Extract DataFrame information
                if isinstance(value, pd.DataFrame):
                    variables[name] = {
                        'type': 'DataFrame',
                        'shape': value.shape,
                        'columns': list(value.columns),
                        'dtypes': value.dtypes.to_dict(),
                        'head': value.head().to_dict()
                    }
                # Extract numeric values
                elif isinstance(value, (int, float, np.number)):
                    variables[name] = {
                        'type': type(value).__name__,
                        'value': float(value)
                    }
                # Extract strings
                elif isinstance(value, str) and len(value) < 1000:
                    variables[name] = {
                        'type': 'str',
                        'value': value
                    }
                # Extract lists/arrays
                elif isinstance(value, (list, tuple, np.ndarray)) and len(value) < 100:
                    variables[name] = {
                        'type': type(value).__name__,
                        'value': list(value) if hasattr(value, '__iter__') else str(value)
                    }

            except Exception as e:
                logger.warning(f"Failed to extract variable {name}: {e}")

        return variables
    
    def validate_code(self, code: str) -> Tuple[bool, str]:
        """Validate code security"""
        # Dangerous keyword check
        dangerous_keywords = [
            'import os', 'import sys', 'import subprocess',
            'exec(', 'eval(', '__import__',
            'open(', 'file(', 'input(',
            'raw_input(', 'reload(',
            'globals(', 'locals()', 'vars(',
            'dir(', 'hasattr(', 'getattr(',
            'setattr(', 'delattr(',
        ]

        code_lower = code.lower()
        for keyword in dangerous_keywords:
            if keyword.lower() in code_lower:
                return False, f"Code contains dangerous operation: {keyword}"

        # Syntax check
        try:
            compile(code, '<string>', 'exec')
            return True, "Code validation passed"
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Code validation failed: {e}"
