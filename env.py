import pathlib


PROJECT_DIR = pathlib.Path.cwd()
DATA_DIR = PROJECT_DIR / 'data'
OUTPUT_DIR = PROJECT_DIR / 'output'


def get_path(relative_path: str) -> pathlib.Path:

    return (
        pathlib.Path(relative_path)
        if relative_path.startswith('C:/')
        else pathlib.Path.joinpath(PROJECT_DIR,relative_path)
    )

dir=get_path('data/adult')
print()
