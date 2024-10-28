import subprocess

# from https://stackoverflow.com/questions/78001716/how-to-get-git-commit-hash-in-a-secure-way-in-python
def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

# from https://stackoverflow.com/questions/78001716/how-to-get-git-commit-hash-in-a-secure-way-in-python
def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

def get_git_status() -> str:
    return subprocess.check_output(['git', 'status']).decode('ascii').strip()

def get_git_diff() -> str:
    return subprocess.check_output(['git', 'diff', 'HEAD']).decode('ascii').strip()


# these logging functions will log important information while printing to stdout 
# for training loss per batch, let's just use tensorboard for that

def init_log(file_path: str):
    global target
    target = open(file_path, "a")
 
def write_to_log(content: str):
    print(content)
    target.write(content + '\n')
