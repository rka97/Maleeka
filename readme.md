# Introduction

Maleeka is an Arabic OCR system whose aim is to recognise scanned computer-produced Arabic text.

# How to run
To run the Python code, install pip and run
    pip install -r requirements.txt
or use Anaconda and install the same packages in requirements.txt.

# Taskplan
The following is done:
1. Basic preprocessing (deskewing, binarizing)
2. Segmenting images into lines.
3. Segmenting lines into words.

What remains:
1. Preprocessing for noise (assuming salt & paper or other common noise forms).
2. Segmenting words into characters (try different approaches, choose the best).
3. Training a model for character classification.
4. Testing the whole process against the ground truth.

## Segmentation
According to Yasser M. Alginahi's A survey on Arabic character segmentation, techniques for segmenting Arabic words into characters generally fall into the following categories:
1. Histogram-based Methods
Old, but seems to be very good for fixed fonts without overlapping characters (the font we have has some of those). TODO.
2. Contour Tracing
Also old, but seem good. TODO.
3. Thinning
Not used alone.
4. Neural Networks
High error rate, not many results. Few good recent ones but they use deep learning (not allowed here). Avoid.

Actually, since deep learning methods seem to work well for this problem, and CNNs can be aproximated well by Bag-of-local-features models (Brendel & Bethge 2019), then maybe we should express using the bag-of-local-features model for both segmentation & recognition in one go (in effect, treating this like an object recognition problem).

5. Graph Theory
Seems to get good rates. TODO. 
6. Morphological operators
Not a lot of results. Avoid.
7. Hidden Markov Models
Very good segmentation accuracy. TODO.
8. Template Matching
Usually costs too much time & is impractical. BUT in our case we have a fixed font and we can use vectorization to speed it up, it may not be so impractical after all. Not sure if avoid or not, we need to discuss it.
9. Transforms
Seems to have a good segmentation accuracy. TODO.
10. Strokes, segments, and tokens
Not a lot of results. Avoid.
 
Methods tried:
1. Latifa et al. (2004) - Histogram-based, did not work well.