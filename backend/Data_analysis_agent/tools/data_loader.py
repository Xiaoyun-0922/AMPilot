"""
Data Loading Tool for Data Analysis Agent

Multi-format data loading tool
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, TYPE_CHECKING
import chardet

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configuration import SUPPORTED_FORMATS, MAX_FILE_SIZE_MB, DATA_DIR

if TYPE_CHECKING:
    from graph.state import DataContext

logger = logging.getLogger(__name__)

class DataLoader:
    """Data loader"""

    def __init__(self):
        """Initialize data loader"""
        self.data_dir = Path(DATA_DIR)
        self.data_dir.mkdir(exist_ok=True)
        logger.info("Data loader initialized successfully")

    def load_data(self, file_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """
        Load data file

        Args:
            file_path: File path
            **kwargs: Additional parameters

        Returns:
            Loading result dictionary
        """
        try:
            file_path = Path(file_path)

            # Check if file exists
            if not file_path.exists():
                return {
                    'success': False,
                    'error': f"File not found: {file_path}",
                    'data': None,
                    'info': {}
                }
            
            # Check file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                return {
                    'success': False,
                    'error': f"File too large: {file_size_mb:.2f}MB > {MAX_FILE_SIZE_MB}MB",
                    'data': None,
                    'info': {}
                }

            # Check file format
            file_ext = file_path.suffix.lower()
            if file_ext not in SUPPORTED_FORMATS:
                return {
                    'success': False,
                    'error': f"Unsupported file format: {file_ext}",
                    'data': None,
                    'info': {}
                }

            # Load data based on file type
            if file_ext == '.csv':
                data = self._load_csv(file_path, **kwargs)
            elif file_ext in ['.xlsx', '.xls']:
                data = self._load_excel(file_path, **kwargs)
            elif file_ext == '.json':
                data = self._load_json(file_path, **kwargs)
            elif file_ext == '.txt':
                data = self._load_text(file_path, **kwargs)
            else:
                return {
                    'success': False,
                    'error': f"Unimplemented file format handler: {file_ext}",
                    'data': None,
                    'info': {}
                }

            # Generate data information
            info = self._generate_data_info(data)

            return {
                'success': True,
                'error': None,
                'data': data,
                'info': info
            }

        except Exception as e:
            logger.error(f"Data loading error: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': None,
                'info': {}
            }
    
    def _load_csv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load CSV file"""
        # Detect encoding
        encoding = self._detect_encoding(file_path)

        # Default parameters
        default_params = {
            'encoding': encoding,
            'index_col': None,
            'header': 0
        }
        default_params.update(kwargs)

        return pd.read_csv(file_path, **default_params)

    def _load_excel(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load Excel file"""
        default_params = {
            'index_col': None,
            'header': 0
        }
        default_params.update(kwargs)

        return pd.read_excel(file_path, **default_params)

    def _load_json(self, file_path: Path, **kwargs) -> Union[pd.DataFrame, Dict, List]:
        """Load JSON file"""
        _ = kwargs  # Suppress unused parameter warning
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Try to convert to DataFrame
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                return pd.DataFrame(data)
        elif isinstance(data, dict):
            # Check if can be converted to DataFrame
            if all(isinstance(v, list) for v in data.values()):
                return pd.DataFrame(data)

        return data
    
    def _load_text(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load text file"""
        encoding = self._detect_encoding(file_path)

        # Try different separators
        separators = ['\t', ',', ';', ' ', '|']

        for sep in separators:
            try:
                df = pd.read_csv(file_path, sep=sep, encoding=encoding, **kwargs)
                if df.shape[1] > 1:  # If successfully split into multiple columns
                    return df
            except:
                continue

        # If all fail, read line by line
        with open(file_path, 'r', encoding=encoding) as f:
            lines = f.readlines()

        return pd.DataFrame({'text': [line.strip() for line in lines]})

    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB

            result = chardet.detect(raw_data)
            encoding = result['encoding']

            # Common encoding mapping
            if encoding and encoding.lower() in ['gb2312', 'gbk']:
                encoding = 'gbk'
            elif encoding and encoding.lower() in ['utf-8-sig']:
                encoding = 'utf-8-sig'
            elif not encoding or result['confidence'] < 0.7:
                encoding = 'utf-8'

            return encoding

        except Exception as e:
            logger.warning(f"Encoding detection failed: {e}, using default encoding utf-8")
            return 'utf-8'
    
    def _generate_data_info(self, data: Union[pd.DataFrame, Dict, List]) -> Dict[str, Any]:
        """Generate data information"""
        info = {}

        if isinstance(data, pd.DataFrame):
            info = {
                'type': 'DataFrame',
                'shape': data.shape,
                'columns': list(data.columns),
                'dtypes': data.dtypes.to_dict(),
                'memory_usage': data.memory_usage(deep=True).sum(),
                'null_counts': data.isnull().sum().to_dict(),
                'numeric_columns': list(data.select_dtypes(include=[np.number]).columns),
                'categorical_columns': list(data.select_dtypes(include=['object', 'category']).columns),
                'sample_data': data.head(3).to_dict()
            }

            # Basic statistical information
            numeric_data = data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                info['statistics'] = numeric_data.describe().to_dict()
        
        elif isinstance(data, dict):
            info = {
                'type': 'dict',
                'keys': list(data.keys()),
                'size': len(data)
            }
        
        elif isinstance(data, list):
            info = {
                'type': 'list',
                'length': len(data),
                'sample': data[:3] if len(data) > 0 else []
            }
        
        return info

    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get basic file information

        Args:
            file_path: File path

        Returns:
            File information dictionary
        """
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                return {
                    'filename': file_path.name,
                    'size_mb': 0,
                    'exists': False,
                    'error': 'File does not exist'
                }

            file_size_mb = file_path.stat().st_size / (1024 * 1024)

            return {
                'filename': file_path.name,
                'size_mb': file_size_mb,
                'exists': True,
                'extension': file_path.suffix.lower(),
                'supported': file_path.suffix.lower() in SUPPORTED_FORMATS
            }

        except Exception as e:
            logger.error(f"Error getting file info: {e}")
            return {
                'filename': str(file_path),
                'size_mb': 0,
                'exists': False,
                'error': str(e)
            }

    def load_data_context(self, file_path: Union[str, Path]) -> 'DataContext':
        """
        Load data context information

        Args:
            file_path: File path

        Returns:
            DataContext object
        """
        from graph.state import DataContext

        try:
            file_path = Path(file_path)

            # Load data
            result = self.load_data(file_path)

            if not result['success']:
                raise Exception(result['error'])

            data = result['data']
            info = result['info']

            # Create DataContext
            if isinstance(data, pd.DataFrame):
                return DataContext(
                    file_path=str(file_path),
                    file_size=file_path.stat().st_size,
                    columns=list(data.columns),
                    shape=data.shape,
                    dtypes={col: str(dtype) for col, dtype in data.dtypes.items()},
                    sample_data=data.head(3).to_dict(),
                    missing_values=data.isnull().sum().to_dict(),
                    summary_stats=info.get('statistics', {})
                )
            else:
                # For non-DataFrame data, create simplified context
                return DataContext(
                    file_path=str(file_path),
                    file_size=file_path.stat().st_size,
                    columns=[],
                    shape=(0, 0),
                    dtypes={},
                    sample_data={'data': str(data)[:200]},
                    missing_values={}
                )

        except Exception as e:
            logger.error(f"Error loading data context: {e}")
            # Return error context
            return DataContext(
                file_path=str(file_path),
                file_size=0,
                columns=[],
                shape=(0, 0),
                dtypes={},
                sample_data={'error': str(e)},
                missing_values={}
            )
