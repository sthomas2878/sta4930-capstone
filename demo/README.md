**System Packages**
- poppler: Required for using pdf2image
    - MacOS: brew installl poppler
- tesseract: Required for using pytesseract
    - MacOS: 
        - brew install tesseract
        - brew install tesseract-lang

**Python Packages**
- requirements.txt

**Opus Dataset (Open Source Parellel Corpora)**
- NewsCommentary v16: Contains ru_en document and sentence aligned translations
    - Used for demo


**On Cosine-Similarity**
- Cosine similarity measures the angle between two vectors in multidimensional space, with each element of the vector representing the count of a word in the document. The cosine-similarity is useful, as it measures the distance irrespective of the individual magnitude (i.e. Vector of [2,3] is the same as vector of [2,7])