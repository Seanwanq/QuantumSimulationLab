import os
import shutil
import cmath


def clean_directory(directory):
    develop_path = os.path.join(os.getcwd(), directory)
    
    if not os.path.isdir(develop_path):
        print(f"Directory {develop_path} does not exist.")
        return
    
    for filename in os.listdir(develop_path):
        file_path = os.path.join(develop_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 删除目录及其所有内容
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
    
    print(f"Cleaned the directory {develop_path}.")

def create_directory(father_directory, directory_name):
    full_path = os.path.join(father_directory, directory_name)
    os.makedirs(full_path, exist_ok=True)
    print(f"Created directory: {full_path}")