import os
import glob

preamble = r"""---
title: "Assignment 6 Plots"
author: "Kulwinder Kaur"
header-includes:
  - \usepackage{amssymb,amsmath}
output: 
  pdf_document
---

"""


class ToMarkdown:
    def __init__(self, directory="") -> None:
        self.directory = directory
        self.files = self.get_files()

    def get_files(self):
        """
        Gets the files in the directory

        Parameters
        ----------
        None

        Returns
        -------
        list
            A list of the files in the directory
        """
        return glob.glob(os.path.join(self.directory, "plots", "*.png"))

    def parse_text(self):
        text_dir = os.path.join(self.directory, "plots", "text.txt")
        text = ""
        with open(text_dir, "r") as f:
            text = f.read()
        texts = text.split("###")
        texts = list(map(lambda x: x.strip(), texts))
        return texts[1:]

    def prepare_text(self):
        texts = self.parse_text()
        len_images = len(self.files)
        len_texts = len(texts)
        for i in range(len_images - len_texts):
            texts.append("")
        return texts

    def write(self):
        title = "Assignment 6"
        header = f"# {title}\n"
        texts = self.prepare_text()
        current_problem = 1

        i = 0
        with open(f"{self.directory}/plots.md", "w") as f:
            f.write(preamble)
            f.write(header)
            f.write(f"\n## Problem {current_problem}\n")
            for file in self.files:
                problem = int(file.split(os.path.sep)[-1][:2])
                if problem != current_problem:
                    current_problem = problem
                    f.write(f"\n## Problem {current_problem}\n")

                text = f"""
![{file}]({file})
{texts[i]}

---
"""
                i += 1
                f.write(text)


if __name__ == "__main__":
    md = ToMarkdown(".")
    # print(md.parse_text())
    md.write()
    md_pdf_command = "pandoc --pdf-engine=xelatex plots.md -o plots.pdf"
    md_pdf_command = r"""pandoc --pdf-engine=xelatex plots.md -o plots.pdf"""
    print("Converting to pdf")
    os.system(md_pdf_command)
    print("Done")
