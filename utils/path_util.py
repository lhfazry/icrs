import os

def ensure_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if os.getcwd() != project_root:
        print(f"Changing working directory to project root: {project_root}")
        os.chdir(project_root)