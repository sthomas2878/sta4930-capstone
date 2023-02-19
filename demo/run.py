## General Imports
import pdb
import os
import shutil

# ## Corus Import (News API) + Selenium/Webscraping Import (For Url Screenshots)
# from corus import load_lenta2
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from chromedriver_py import binary_path

## PDF Generation
import random
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import letter
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, cm
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

## OCR Imports
from pdf2image import convert_from_path
import cv2
import pytesseract

## Language Processing 
import nltk

# The engine for sentence/word tokenization. which uses an unnsupervised algorithm to parse for tokenization. Create a folder named 'nltk_data' in your env/lib directory, and download punkt to it 
nltk.download('punkt', download_dir='/Users/nicolastobon/Desktop/sta4930-capstone/translation/models/fairseq/fairseq macOS/env/lib/nltk_data')

# The stopwords used to parse out text that isn't relevant to a word ('.', '?', etc.)
nltk.download('stopwords', download_dir='/Users/nicolastobon/Desktop/sta4930-capstone/translation/models/fairseq/fairseq macOS/env/lib/nltk_data')

from cer import calculate_cer, calculate_cer_corpus

# OCR Analysis
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.cluster import cosine_distance
import pandas as pd

## Translation Model Imports
import torch
import numpy as np


# # Downloading our data from Corus  
# data_path = os.path.join(proj_path, 'translation/models/fairseq/fairseq macOS/demo/data/', 'lenta-ru-news.csv.bz2')
# records = load_lenta2(path)

# options = webdriver.ChromeOptions()
# options.add_argument('--headless')


# service_object = Service(binary_path)
# driver = webdriver.Chrome(service=service_object, options=options)
# driver.set_window_size(2048, 2000) # set the window size

# for i in range(0, 10):
#     try: 
#         record = next(records)
#         url = record.url
#         file = os.path.join(proj_path, 'translation/models/fairseq/fairseq macOS/demo/data/images', f'{i}.png')

#         driver.get(url) # load the web page
#         driver.save_screenshot(file) # save a screenshot
#         driver.save_scr
#         print("Webpage screenshot saved as", file)
#     except Exception as e:
#         print(e)

def create_pdf(input_file_path, output_file_path):
    
    ## Documents require a Canvas to write on
    def page(canvas, doc):
        canvas.saveState()
        canvas.restoreState()

    try: 
        with open(input_file_path, 'r') as f:
            lines = f.readlines()
     
        styles = getSampleStyleSheet() # Initialize sample style class
        pdfmetrics.registerFont(TTFont('DejaVuSerif', '/Users/nicolastobon/Library/Fonts/DejaVuSerif/DejaVuSerif.ttf', 'UTF-8')) # Register Font

        doc = SimpleDocTemplate(output_file_path, pagesize=A4,
                        rightMargin=2*cm,leftMargin=2*cm,
                        topMargin=2*cm,bottomMargin=2*cm) # Set our document template to A4 (letter size)

        Story = [Spacer(1, 0.2*(inch))] # Create an initial spacing
        style = styles['Normal'] # Assign normal style
        style.fontName = 'DejaVuSerif' # Assign fontname
        style.fontSize = 4 # Assign fontSize

        for line in lines: 
            p = Paragraph(line, style) # Create a paragraph
            Story.append(p) # Append a paragraph
            #Story.append(Spacer(1, 0.2*inch)) ## Append an optional aextra space
        
        doc.build(Story, onFirstPage=page, onLaterPages=page) ## Docs are built progressively through Stores on page settings 

    except Exception as e:
        raise(e)

    return output_file_path

def pdf_to_images(file_path, articles_dir): 

    try: 
        images = convert_from_path(file_path) # Convert pdf into image
        print(f'Converting pdf file from {file_path}, to images')

        image_folder_dir = os.path.join(articles_dir, 'images')
        os.mkdir(image_folder_dir)

        for page in range(len(images)): 
                    image_dir = os.path.join(image_folder_dir, f'page_{str(page)}.png')
                    images[page].save(image_dir, 'PNG') # Save pages as images in the pdf
                    print(f'Saved {image_dir}')
    except Exception as e:
        raise(e)

    return image_folder_dir

def images_to_text(images, articles_dir):

    text = []
    try: 
        custom_config = r'--psm 6'
        for ind in reversed(os.listdir(images)): 
            img = cv2.imread(os.path.join(images, ind))
            text.append(pytesseract.image_to_string(img, lang='rus', config=custom_config))
            print(f'The length of the text is now at: {len(text)}')

        ru_OCR_path = str(os.path.join(articles_dir, 'ru_OCR.txt')) # ru_OCR is the non-processed OCR 
        ru_OCR_P_path = str(os.path.join(articles_dir, 'ru_OCR_P.txt')) # ru_OCR_P is the OCR sent to preprocessing
        with open(ru_OCR_path, 'w') as file:
            for page in text:
                file.write(page)
        with open(ru_OCR_P_path, 'w') as file:
            for page in text:
                file.write(page)

    except Exception as e:
        raise(e)

    return ru_OCR_P_path # Return the OCR for pre-processing

def preprocess(file_path, tag): 

    text_p = []
    try:
        if not tag:
            with open(file_path, 'r') as file:
                text = file.readlines()

            cur_str = f''

            for ind, line in enumerate(text):
                # Append cur_str, reset cur_str
                if line == '\n': 
                    cur_str = cur_str.strip()+'\n' 
                    text_p.append(cur_str)
                    cur_str = f''
                    continue
                else:
                    cur_str += line.strip().split('\n')[0] + ' ' # Append string, if not '\n'

            ## Lines should have a space at the end, as the append method will strip the last space

            text_p.append(cur_str.strip()) # Last string in the file

            with open(file_path, 'w') as file:
                file.writelines(text_p)

        else: 
            with open(file_path, 'r') as file:
                text = file.readlines()

            for ind, line in enumerate(text):
                # Skip lines that are not text
                if line == '<HEADLINE>\n' or line == '<P>\n' or line == '\n':
                    continue
                else:
                    text_p.append(line)
    
            with open(file_path, 'w') as file:
                file.writelines(text_p)

    except Exception as e:
        raise(e)

    return file_path

def ocr_analysis(file_path_1, file_path_2):

    try:
        with open(file_path_1, 'r') as file:
            text_1 = file.read()

        with open(file_path_2, 'r') as file:
            text_2 = file.read()

        # Tokenize the documents into lists of words and remove stop words
        stop_words = set(stopwords.words('russian'))
        text_1_words = [word for word in word_tokenize(text_1.lower()) if word.isalnum() and word not in stop_words]
        text_2_words = [word for word in word_tokenize(text_2.lower()) if word.isalnum() and word not in stop_words]

        # Calculate the frequency distribution of words in the documents
        doc1_freq = FreqDist(text_1_words)
        doc2_freq = FreqDist(text_2_words)

        # Combine the words into a set
        words = set(doc1_freq.keys()).union(set(doc2_freq.keys()))

        # Convert the frequency distributions into vectors
        doc1_vec = [0 if word not in doc1_freq else doc1_freq[word] for word in words]
        doc2_vec = [0 if word not in doc2_freq else doc2_freq[word] for word in words]
        
        assert(len(doc1_vec) == len(doc2_vec))

        # Calculate the cosine distance between the two vectors
        cosine_score = cosine_distance(doc1_vec, doc2_vec)    

        ''' 
        The cosine distance is 1-cos(dot product/norm product) and lies between [0,2]
        Values close to 0 signify close similarity
        Values close to 1 signify no similarity
        Values close to 2 signify close opposite
        '''
        print("Cosine distance between the two documents:", cosine_score)
        
        return cosine_score

    except Exception as e:
        raise(e)


def get_text(file_path): 

    try:
        with open(file, 'r') as file:
            text = file.readlines()

    except Exception as e:
        raise(e)

    return text


def remove_dir(dir):
    for sub_dir in os.listdir(dir):
        shutil.rmtree(os.path.join(dir, sub_dir))



## Establish project path and tesseract path
proj_path = os.path.abspath('/Users/nicolastobon/Desktop/sta4930-capstone')
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

## Define outer paths
articles = os.path.join(proj_path, 'translation/models/fairseq/fairseq macOS/demo/data/articles')
en_g = os.path.join(proj_path, 'translation/models/fairseq/fairseq macOS/demo/data/NewsCommentary_en-ru.txt/split/en')
ru_g = os.path.join(proj_path, 'translation/models/fairseq/fairseq macOS/demo/data/NewsCommentary_en-ru.txt/split/ru')

## Create articles directory
if not os.path.isdir(articles):
    os.mkdir(articles)

corpus_sim_scores = []

file_index = [file.split('.txt')[0] for file in np.random.choice(os.listdir(ru_g), 10)] # Generates random indices from the ru_dir, and uses the indices to retrieve those articles

for ind in file_index:


    try:

        if os.path.isfile(os.path.join(ru_g, f'{ind}.txt')):

            article = os.path.join(articles, ind)
            pdf_path = os.path.join(article, f'{ind}.pdf')

            ## Create folder documents 
            if not os.path.isdir(article):
                os.mkdir(article)

            ## Create en, ru paths for txt files
            en_l = os.path.join(article, 'en')
            ru_l = os.path.join(article, 'ru')

            ## Copy over the english text and russian text into data directory
            shutil.copy(os.path.join(en_g, f'{ind}.txt'), en_l)
            shutil.copy(os.path.join(ru_g, f'{ind}.txt'), ru_l) 
            print(f'Created the directory for {ind}')
        
            pdf = create_pdf(ru_l, pdf_path) # Create pdf by randomizing txt file, returns pdf directory
            print(f'Randomized the .txt file, converted into pdf')

            images = pdf_to_images(pdf, article) # Create images from pdf, returns image folder directory

            ru_t_l = images_to_text(images, article) # Create ruT.txt from OCR on images, returns ruT.txt path

            # Pre-processes and formats en_l, ru_l, and ru_t_l, parsing on delimiters 
            en_l = preprocess(en_l, True)
            ru_l = preprocess(ru_l, True)
            ru_t_l = preprocess(ru_t_l, False)

            ocr_res = ocr_analysis(ru_l, ru_t_l) # OCR Analysis on ru_t and ru_t_l, returns corpus cosine-similarity score

            corpus_sim_scores.append(ocr_res) # Needs revisions

            #en_text = get_text(en_l)
            #ru_text = get_text(ru_l)

            #model = torch.hub.load(repo_or_dir='pytorch/fairseq', model='transformer.wmt19.ru-en', checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt', source='github', tokenizer = 'moses', bpe = 'fastbpe')

            # rand_ind = np.random.choice(np.arange(min(len(ru), len(en))))
            # ru_line = ru_text[rand_ind]

            # t_line = model.translate(ru_line)
            # print(f'The line {ru_line} has been translated into {t_line}')

            # #bleu_score = nltk.bleu([t_line], en[rand_ind])
            # #print(f'Bleu Score is {bleu_score}')

    except Exception as e:
        raise(e)


print(pd.Series(corpus_sim_scores).describe())
#shutil.rmtree(articles_dir) # Removes the articles_directory, useful for resetting the script


