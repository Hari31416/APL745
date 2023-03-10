import zipfile
import os


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
    # ziph is zipfile handle
    for root, _, files in os.walk(path):
        for file in files:
            if file not in exclude:
                ziph.write(os.path.join(root, file))


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
    excludes = [".ipynb_checkpoints", "zip.py", zipname]
    main(".", zipname, excludes)
