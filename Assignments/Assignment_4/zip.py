import zipfile
import os
from style import cprint


def zipdir(path, ziph, exclude=[]):
    """
    Creates a zip file of the directory

    Parameters
    ----------
    path : str
        The path to the directory to zip
    ziph : zipfile.ZipFile
        The zipfile handle
    exclude : list, optional
        A list of files to exclude from the zip file

    Returns
    -------
    None
    """
    dirs_to_exclude = [f"{path}/{d}" for d in exclude]
    for root, _, files in os.walk(path):
        if root in dirs_to_exclude:
            cprint(f"Excluding Directory {root} ...", "red")
            continue
        for file in files:
            if file not in exclude:
                cprint(f"Adding File {os.path.join(root, file)} ...", "green")
                ziph.write(os.path.join(root, file))
            else:
                cprint(f"Excluding File {file} ...", "warning")


def zipit(path, zipname, exclude=[]):
    """
    Zips the directory

    Parameters
    ----------
    path : str
        The path to the directory to zip
    zipname : str
        The name of the zip file
    exclude : list, optional
        A list of files to exclude from the zip file

    Returns
    -------
    None
    """
    with zipfile.ZipFile(zipname, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipdir(path, zipf, exclude)


def main(dir, zipname, exclude=[]):
    zipit(dir, zipname, exclude)


if __name__ == "__main__":
    zipname = "2021PHS7181.zip"
    excludes = [
        ".ipynb_checkpoints",
        "zip.py",
        zipname,
        "__pycache__",
        "image_to_md.py",
        "plots.md",
        "text.txt",
        "NN_2",
        "NN_2/nn",
        "NN_2/nn/__pycache__",
        "nn/__pycache__",
        "Neural_Networks.ipynb",
        "Neural_Networks.pdf",
        "README.md",
    ]
    main(os.curdir, zipname, excludes)
