import inspect
import os
from typing import Any, Optional, Dict

def nameof(var: Any) -> Optional[str]:
    caller_globals: Dict[str, Any] = inspect.stack()[1].frame.f_globals
    for name, value in caller_globals.items():
        if value is var:
            return name
    return None

def clean_directory(directory: str) -> None:
    develop_path: str = os.path.join(os.getcwd(), directory)
    if not os.path.exists(develop_path):
        print(f"Directory {develop_path} does not exist.")
        return
    for filename in os.listdir(develop_path):
        file_path: str = os.path.join(develop_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
    print(f"Cleaned the directory {develop_path}.")

