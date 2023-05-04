# evaluate.py directory
#  - png's
#  - ocr
#  - translation
#  - sentiment_analysis

import sys
import os

# OCR Imports
from pdf2image import convert_from_path
import cv2
import pytesseract
import os
import Levenshtein as lev

# Translation Imports
import pandas as pd

# Sentiment Analysis Imports
import pysentimiento as ps 

def ocr(file, file_path, temp_dir):

    def clean_ocr_string(text):
        #Replace all new lines and carriage returns with spaces or nothing
        cleanNewLine = text.replace('\n', ' ').replace('\r', '')
        #cleanPunct = cleanNewLine.replace('?', '.').replace('...', '.')
        return cleanNewLine

    try:
        # Create temp output path
        temp_output = f'{os.path.join(temp_dir, "ocr")}'
        if not os.path.exists(temp_output):
            os.makedirs(temp_output)

        output_path = f'{os.path.join(temp_output, file[:-4])}.txt'
        img = cv2.imread(file_path)
        
        #Read image with pytesseract-ocr, output is a long string
        text = pytesseract.image_to_string(img, lang='rus')
        cleanString = clean_ocr_string(text)
        
        #Create sentence array 
        ocrSentenceArr = cleanString.split('.')
        del(ocrSentenceArr[-1])

        while("" in ocrSentenceArr):
            ocrSentenceArr.remove("")

        with open(output_path, 'w') as f:
            for sentence in ocrSentenceArr:
                # Remove space from front of ocr_line
                if len(sentence) > 2:
                    while sentence[0] == ' ':
                        sentence = sentence[1:]
                f.write(sentence + '.\n') 
    except Exception as e:
        raise(e)
    
def get_data(ocr_dir):

    try: 
        data_dict = {file:{} for file in os.listdir(ocr_dir)}

        for file in data_dict.keys():
            with open(os.path.join(ocr_dir, file), 'r') as f:
                text = f.readlines()
            data_dict[file]['ocr'] = text

        # Convert the dictionary to a pandas dataframe
        data = pd.DataFrame.from_dict(data_dict, orient='index')
    except Exception as e:
        raise(e)

    return data

    
def translate(data, temp_dir):

    def get_model():
        from transformers import FSMTForConditionalGeneration, FSMTTokenizer, FSMTConfig, PreTrainedTokenizer, PreTrainedModel 
        tokenizer = FSMTTokenizer.from_pretrained("facebook/wmt19-ru-en")
        config= FSMTConfig().from_pretrained("facebook/wmt19-ru-en")
        config.max__new_tokens = 512
        model = FSMTForConditionalGeneration.from_pretrained("facebook/wmt19-ru-en")
        return (model, tokenizer)
    
    def apply_translation(input, model, tokenizer):
        translations = []
        try: 
            for line in input:
                input_tokens = tokenizer([line], is_split_into_words=False, return_tensors = 'pt', padding = True).input_ids
                outputs = model.generate(input_tokens)
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                translations.append(decoded[0])
            return translations
        except Exception as e:
            print(e)
    
    # # Create temp output path
    # temp_output = f'{os.path.join(temp_dir, "translation")}'
    # if not os.path.exists(temp_output):
    #     os.makedirs(temp_output)

    try:
        model, tokenizer = get_model()
        data['translation'] = data.apply(lambda x: apply_translation(x['ocr'], model=model, tokenizer=tokenizer), axis=1)
    except Exception as e:
        raise(e)

    return data

def analyze_sentiment(data):

    def sentiment_to_score(data): 
        data['overall_sentiment_score'] = data['overall_sentiment'].apply(lambda x: [1 if line_score == 'POS' else (-1 if line_score == 'NEG' else 0) for line_score in x])
        data['overall_sentiment_score'] = data['overall_sentiment_score'].apply(lambda x: sum(x)/len(x))

    def analyze(text, analyzers):

        try: 
            analyses = []
            for analyzer in analyzers:
                analysis = [analyzer.predict(line).output for line in text]
                analyses.append(analysis)
        except Exception as e:
            raise(e)
        
        return tuple(analyses)
        

    try:  
        analyzer = ps.create_analyzer(task="sentiment", lang="en")
        emotion_analyzer = ps.create_analyzer(task="emotion", lang="en")
        hate_speech_analyzer = ps.create_analyzer(task="hate_speech", lang="en")
        irony_analyzer = ps.create_analyzer(task="irony", lang="en")
        
        analyzers = [analyzer, emotion_analyzer, hate_speech_analyzer, irony_analyzer]

        data[['overall_sentiment', 'emotion', 'hate', 'irony']] = data.apply(lambda x: analyze(x['translation'], analyzers), axis=1, result_type='expand')
        sentiment_to_score(data)
    except Exception as e:
        raise(e)
    
    return data

    
# python evaluate.py absolute_directory
if __name__ == "__main__":
    # Get arguments from command line

    directory = sys.argv[1]

    if not os.path.isdir(directory):    
        raise(Exception("Please enter a valid directory"))
    
    # Create temp directory
    temp_dir = os.path.join(directory, "temp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # Get all files in directory
    files = os.listdir(directory)

    # Path of Tesseract executable | MacOS Homebrew: brew install tesseract  
    pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

    for file in files:
        if not file.endswith('.png'):
            continue
        else:
            file_path = os.path.join(directory, file)
            ocr(file, file_path, temp_dir)
            
    # Translation
    data = get_data(os.path.join(temp_dir, "ocr"))
    data = translate(data, temp_dir)

    # Sentiment Analysis
    data = analyze_sentiment(data)

    data.to_pickle(os.path.join(directory, "evaluation.pkl"))