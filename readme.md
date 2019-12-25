# Introduction

Maleeka is an Arabic OCR system whose aim is to recognise scanned computer-produced Arabic text.

# Implemented Method
This system implements the Holistic Arabic OCR technique of [Nashwan et al. (2017)](https://www.mdpi.com/2313-433X/4/1/6). For a detailed description, please see the report in the docs/ folder. Training on about 600 files each having one or two paragraphs of Arabic text, we can obtain on a test set of sixty files an average edit distance accuracy of 91-94% with a running time of about 3 seconds per file.

# How to run
To run the Python code, install pip and run
    pip install -r requirements.txt
or use Anaconda and install the same packages in requirements.txt. You either need a pre-trained model or a dataset of annotated images. Place the dataset/model in the directories indicated the code (or alter the code's directories).
