import os
import yaml
import random
import numpy as np
import pandas as pd
from utils.utils import Timer, get_logger
import timeit
import torch
import rouge
import sacrebleu
import nltk
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd, log


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
import os
import Levenshtein as lev

class Dataset:
    pass

def take_input() -> str: 
    user_prompt = []
    while True:
        line = input()
        if line:
            user_prompt.append(line)
        else:
            break
    return "\n".join(user_prompt)

def create_temp_dir(cfg, logger): 
    ## Check if the temp directory exists
    try:
        path = get_original_cwd()
        if not os.path.exists(path + '/temp'):
            os.makedirs(path + '/temp')
        return os.path.join(path, 'temp')
    except Exception as e:
        raise(e)


def create_pdfs(cfg:DictConfig, logger):
    
    ## Documents require a Canvas to write on
    def page(canvas, doc):
        canvas.saveState()
        canvas.restoreState()

    path = get_original_cwd()
    pdf_folder = os.path.join(temp_dir, 'pdf', cfg.create_pdf.name)
    txt_folder = os.path.join(path, "datasets", "demo", cfg.create_pdf.name)

    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)

    try: 
        with Timer('PDF Generation', logger) as pdf_gen:
            for file in os.listdir(txt_folder):
                    logger.info(f'Generating PDF for file: {file}')
                    file_path = os.path.join(txt_folder, file)
                    temp_file_path = f'{os.path.join(pdf_folder, file[:-4])}.pdf'
            
                    with open(file_path, encoding='utf-8') as f:
                        lines = f.readlines()
                
                    styles = getSampleStyleSheet() # Initialize sample style class
                    pdfmetrics.registerFont(TTFont('DejaVuSerif', os.path.join(path, "fonts", "DejaVuSerif", 'DejaVuSerif.ttf'), 'UTF-8')) # Register Font

                    doc = SimpleDocTemplate(temp_file_path, pagesize=A4,
                                    rightMargin=2*cm,leftMargin=2*cm,
                                    topMargin=2*cm,bottomMargin=2*cm) # Set our document template to A4 (letter size)

                    Story = [Spacer(1, 0.2*(inch))] # Create an initial spacing
                    style = styles['Normal'] # Assign normal style
                    style.fontName = 'DejaVuSerif' # Assign fontname
                    style.fontSize = 12 # Assign fontSize

                    for line in lines: 
                        p = Paragraph(line, style) # Create a paragraph
                        Story.append(p) # Append a paragraph
                        #Story.append(Spacer(1, 0.2*inch)) ## Append an optional exttra space
                    
                    doc.build(Story, onFirstPage=page, onLaterPages=page) ## Docs are built progressively through Stores on page settings ​
    except Exception as e:
        raise(e)
    
def convert_pdf(cfg:DictConfig, logger): 

    pdf_folder = os.path.join(temp_dir, 'pdf', cfg.create_pdf.name)
    png_folder = os.path.join(temp_dir, 'png', cfg.create_pdf.name)

    if not os.path.exists(png_folder):
        os.makedirs(png_folder)

    try: 
        with Timer("PNG Generation", logger) as png_gen:
            for file in os.listdir(pdf_folder): 
                logger.info(f'Generating PNG for file: {file}')
                file_path = os.path.join(pdf_folder, file)
                temp_file_path = f'{os.path.join(png_folder, file[:-4])}.png'    

                images = convert_from_path(file_path)
                images[0].save(temp_file_path)  
    except Exception as e:
        raise(e)

def ocr(cfg:DictConfig, logger):

    def clean_ocr_string(text):
        #Replace all new lines and carriage returns with spaces or nothing
        cleanNewLine = text.replace('\n', ' ').replace('\r', '')
        #cleanPunct = cleanNewLine.replace('?', '.').replace('...', '.')
        return cleanNewLine
    
    # Caclulate the Levenshtein Distance between two strings, outputs calculation to dataframe
    def ocrDistanceCal(ocrSentenceArr, reference_folder, file, out_df):

        path = get_original_cwd()

        # ocr_file_path = os.path.join(ocr_folder, file)
        reference_file_path = f'{os.path.join(path, reference_folder, file[:-4])}.txt'
        logger.info(f'Calculating Levenshtein Distance for file: {file[:-4]}')

        i = 0
        levyDist = []
        charCountArr = []
        lineNum = []
        boolPerfectAlignment = False
        with open(reference_file_path, encoding='utf-8') as txtFile:
            lineCount = 0
            for line in txtFile:
                #Replace question marks with periods, and triple periods to match clean ocr string 
                line = line.replace('?', '.').replace('...', '.').replace('\n', '')
                #When ocr split into strings removes period, -> take period away from end of line
                
                while(line[-1] == ' ' or line[-1] == '.' or line[-1] == '\n'):
                    line = line[:-1]
                #Remove weird space from front of ocrLine
                if len(ocrSentenceArr[i]) > 10:
                    while ocrSentenceArr[i][0] == ' ':
                        ocrSentenceArr[i] = ocrSentenceArr[i][1:]
                            
                #Clean ocr line: Remove space before 
                dist = lev.distance(line, ocrSentenceArr[i])
                i = i+1
                charCount = len(line)
                levyDist.append(dist)
                charCountArr.append(charCount)
                lineCount += 1
            
            lineNum = list(np.arange(1,lineCount+1, 1))

            file_list = np.repeat(file, lineCount+1)

            out_tup = zip(file_list, lineNum, levyDist, charCountArr)
            temp_df = pd.DataFrame(out_tup, columns=['file', 'line_num', 'levDist', 'charCount'])
            out_df = pd.concat([out_df, temp_df], ignore_index=True)

        return out_df
    
    # Create folders/path for OCR Input and Output
    input_folder = os.path.join(get_original_cwd(), cfg.ocr.input_path)
    output_folder = os.path.join(temp_dir, 'ocr', cfg.ocr.name)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create dataframe for OCR Levenshtein Distance
    if cfg.ocr.lev_dist == True:
        out_df = pd.DataFrame(columns=['line_num', 'levDist', 'charCount'])


    try:
        with Timer ('OCR Generation', logger) as ocr_gen: 
            for file in os.listdir(input_folder):
                logger.info(f'Generating OCR for file: {file}')

                input_file = os.path.join(input_folder, file)
                output_path = f'{os.path.join(output_folder, file[:-4])}.txt'

                img = cv2.imread(input_file)
                
                #Read image with pytesseract-ocr, output is a long string
                text = pytesseract.image_to_string(img, lang='rus')
                cleanString = clean_ocr_string(text)
                
                #Create sentence array 
                ocrSentenceArr = cleanString.split('.')
                del(ocrSentenceArr[-1])

                while("" in ocrSentenceArr):
                    ocrSentenceArr.remove("")

            
                if cfg.ocr.lev_dist == True:
                    out_df = ocrDistanceCal(ocrSentenceArr, cfg.ocr.input_path, file, out_df)

                with open(output_path, 'w') as f:
                    for sentence in ocrSentenceArr:
                        if len(sentence) > 2:
                            while sentence[0] == ' ':
                                sentence = sentence[1:]
                        f.write(sentence + '.\n') 
                
            if cfg.ocr.lev_dist == True:
                lev_path = os.path.join(temp_dir, 'ocr', 'calculation')
                out_df.to_csv(lev_path, index=False)
            
    except Exception as e:
        raise(e)


def get_data(cfg:DictConfig): 

    try: 

        path = get_original_cwd()

        # If the data is already in the temp folder, load it from there
        if 'ocr' in cfg.keys():
            data_folder = os.path.join(path, 'temp', 'ocr', cfg.ocr.name)
            data_dict = {file:{} for file in os.listdir(data_folder)}

        else: 
            data_folder = os.path.join(path, cfg.translation_data.ocr_path)
            assert(cfg.translation_data.samples <= len(os.listdir(data_folder)))
            sampled_files = random.sample(os.listdir(data_folder), cfg.translation_data.samples)
            data_dict = {file:{} for file in sampled_files}


        for file in data_dict.keys():
            with open(os.path.join(data_folder, file), 'r') as f:
                text = f.readlines()
            data_dict[file]['source'] = text

        if 'source_path' in cfg.translation_data.keys():

            # Ensure that translation_data is in the config file
            assert('score' in cfg.keys())
            source_data_folder = os.path.join(path, cfg.translation_data.ocr_path)
            for file in data_dict.keys():
                with open(os.path.join(source_data_folder, file), 'r') as f:
                    text = f.readlines()
                data_dict[file]['reference'] = text

            # Convert the dictionary to a pandas dataframe
        data = pd.DataFrame.from_dict(data_dict, orient='index')
        
        # elif (cfg.data.name == 'tweets'): 
        #     assert (cfg.data.samples <= 100)
        #     # Get the tweets data
        #     data = pd.read_csv(f'{path}/{cfg.data.path}', header = 0, names = ['source', 'sentiment', 'id', 'tag']).sample(cfg.data.samples)

    except Exception as e:
        raise(e)

    return data

def translate(input:list[str], cfg:DictConfig, model, row, logger):
   
    try: 
        translations = []
        times = []
        if cfg.translation_model.model_hub == 'torch':
            if cfg.translation_data.name == 'tweets': 
                with Timer(f'Translation of Row: {row}', logger) as translator:
                    output = model[0].translate([input], axis=1)
                times.append(translator.elapsed_time)
                translations.append(output[0])
            else: 
                translations = []
                times = []
                for ind,line in enumerate(input):
                    with Timer(f'Translation of Row: {row} | Line: {ind}', logger) as translator:
                        output = model[0].translate([line], axis=1)
                    times.append(translator.elapsed_time)
                    translations.append(output)

            return (translations, times)
        elif cfg.translation_model.model_hub == 'huggingface': 
            if cfg.translation_data.name == 'tweets': 
                with Timer(f'Translation of Row: {row}', logger) as translator:
                    input_tokens = model[1]([input], is_split_into_words=False, return_tensors = 'pt', padding = True).input_ids
                    outputs = model[0].generate(input_tokens)
                    decoded = model[1].batch_decode(outputs, skip_special_tokens=True)
                times.append(translator.elapsed_time)
                translations.append(decoded[0])
            else: 
                translations = []
                times = []
                for ind, line in enumerate(input):
                    with Timer(f'Translation of Row: {row} | Line: {ind}', logger) as translator:
                        input_tokens = model[1]([line], is_split_into_words=False, return_tensors = 'pt', padding = True).input_ids
                        outputs = model[0].generate(input_tokens)
                        decoded = model[1].batch_decode(outputs, skip_special_tokens=True)
                    times.append(translator.elapsed_time)
                    translations.append(decoded[0])
            return (translations, times)
        else: 
            raise (f'Translation for {row} unable to be completed')
        
    except Exception as e:
        raise(e)

def translate_file(data:pd.DataFrame, cfg:DictConfig, model, logger):

    # Translate the source data
    try: 
        if cfg.translation_model.model_hub == 'torch':
            data['translation'] = data.apply(lambda x: translate(x['source'], cfg=cfg, model=model, row=x.name, logger=logger), axis=1)
            return data
        elif cfg.translation_model.model_hub == 'huggingface':
                #  Run the translation function on each row of the dataframe and store the results (translation, time) in two separate columns
                data['translation'] = data.apply(lambda x: translate(x['source'], cfg=cfg, model=model, row=x.name, logger=logger), axis=1)
        else:  
            raise(f'Translation for {cfg.translation_model.model_hub} not supported')
        
        # Split the translation column into two columns, one for the translation and one for the time
        data[[f'translation', 'time']] = pd.DataFrame(data[f'translation'].tolist(), index=data.index)

    except Exception as e:
        raise(e)

    return data

def calculate_scores(data:pd.DataFrame, cfg:DictConfig, logger):

    # Calculate the BLEU Score
    # with Timer('Initailizing BLEU', logger):
    #     bleu = sacrebleu.metrics.bleu.BLEU(
    #         lowercase = False,
    #         force = True,
    #         tokenize = 'none',
    #         smooth_method = 'exp',
    #         smooth_value = None,
    #         max_ngram_order = 4,
    #         effective_order = False,
    #     )

    # Calculate the BLEU Score

    if "BLEU" in cfg.translation_score.metrics:

        with Timer('Calculating BLEU', logger):
            '''
            The BLEU score is calculated using the nltk.translate.bleu_score function with the default parameters: 
            epsilon = 0.1
            alpha = 5
            k = 5
            '''
            chencherry = nltk.translate.bleu_score.SmoothingFunction()
            data['bleu'] = data.apply(lambda x: [nltk.translate.bleu_score.sentence_bleu(x['translation'][ind], x['reference'][ind], smoothing_function=chencherry.method7) 
                                                for ind in np.arange(len(x['reference']))] if (len(x['reference']) == 10 and len(x['translation']) == 10) else None, axis=1)
            # data['BLEU'] = data.apply(lambda x: bleu.corpus_score(x['translation'], x['reference']), axis=1)

    # Calculate the TER Score
    if "TER" in cfg.translation_score.metrics:

        with Timer('Initializing TER', logger):
            ter = sacrebleu.metrics.ter.TER(
                    normalized = False,
                    no_punct = True,
                    asian_support = False,
                    case_sensitive = False
                )
        with Timer('Calculating TER', logger):
            data['ter'] = data.apply(lambda x: [ter.sentence_score(x['translation'][ind], [x['reference'][ind]]).score
                                                for ind in np.arange(len(x['reference']))] if (len(x['reference']) == 10 and len(x['translation']) == 10) else None, axis=1)
            
    # Calculate the CHRF Score
    if "CHRF" in cfg.translation_score.metrics:
            
        with Timer('Initializing CHRF', logger):
            chrf = sacrebleu.metrics.chrf.CHRF(
                    char_order = 6,
                    word_order = 0,
                    beta = 2,
                    lowercase = False,
                    whitespace = False,
                    eps_smoothing = False
                )
        with Timer('Calculating CHRF', logger):
            data['chrf'] = data.apply(lambda x: [chrf.sentence_score(x[f'translation'][ind], [x['reference'][ind]]).score
                                                for ind in np.arange(len(x['reference']))] if (len(x['reference']) == 10 and len(x['translation']) == 10) else None, axis=1)

    # # Calculate the ROUGE Score
    # rouge_scorer = rouge.Rouge(
    #     metrics = ['rouge-l'],
    #     return_lengths = False,
    #     raw_results = False,
    #     exclusive = False
    # )
    # data['ROUGE'] = data.apply(lambda x: rouge_scorer.get_scores(x['translation'], x['reference']), axis=1, ignore_empty=True)
    # COMET Score, BERT Score, and ROUGE Score are not implemented yet

    return data

def get_model(cfg:DictConfig, logger):

    if cfg.translation_model.model_hub == 'torch':
        try: 
            models = {x for x in torch.hub.list('pytorch/fairseq')}
            assert(cfg.version.name in models)
        except Exception as e:
            raise(e)
        
        if {cfg.version.ensemble == True}:
            model = torch.hub.load(repo_or_dir='pytorch/fairseq', model=cfg.translation_model.name, verbose=False,
            checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt', source='github', tokenizer = 'moses', bpe = 'fastbpe')
        else: 
            model = torch.hub.load(repo_or_dir='pytorch/fairseq', model={cfg.translation_model.name + '.single_model'}, verbose=False,
            source='github', tokenizer = 'moses', bpe = 'fastbpe')

        log.info(f'Model_Config:\n{(OmegaConf.to_yaml(model.cfg))}')

    elif cfg.translation_model.model_hub == 'huggingface':
        from transformers import FSMTForConditionalGeneration, FSMTTokenizer, FSMTConfig, PreTrainedTokenizer, PreTrainedModel 
        tokenizer = FSMTTokenizer.from_pretrained(cfg.translation_model.name)
        config= FSMTConfig().from_pretrained(cfg.translation_model.name)
        config.max__new_tokens = 512
        model = FSMTForConditionalGeneration.from_pretrained(cfg.translation_model.name)

        log.info(f'Model_Config:\n{yaml.dump(model.config, default_flow_style=False)}')

    else: 
        raise('Model Hub not found, please check your config file.')
    
    return (model, tokenizer)

def analyze_sentiment(data, cfg:DictConfig, logger):

    def sentiment_to_score(data): 
        data['overall_sentiment_score'] = data['overall_sentiment'].apply(lambda x: [1 if line_score == 'POS' else (-1 if line_score == 'NEG' else 0) for line_score in x])
        data['overall_sentiment_score'] = data['overall_sentiment_score'].apply(lambda x: sum(x)/len(x))
        

    try: 
        with Timer('Initializing Sentiment Analyzers', logger):
            import pysentimiento as ps 
            analyzer = ps.create_analyzer(task="sentiment", lang="en")
            emotion_analyzer = ps.create_analyzer(task="emotion", lang="en")
            hate_speech_analyzer = ps.create_analyzer(task="hate_speech", lang="en")
            irony_analyzer = ps.create_analyzer(task="irony", lang="en")

        
        analyzers = [analyzer, emotion_analyzer, hate_speech_analyzer, irony_analyzer]

        data[['overall_sentiment', 'emotion', 'hate', 'irony']] = data.apply(lambda x: analyze(x[f'translation'], analyzers, x.name, logger), axis=1, result_type='expand')
        sentiment_to_score(data)

    except Exception as e:
        raise(e)
    
    return data

def analyze(text, analyzers, row, logger):

    try: 
        with Timer(f'Analyzing Sentiment Tasks for {row}', logger):
            analyses = []
            for analyzer in analyzers:
                analysis = [analyzer.predict(line).output for line in text]
                analyses.append(analysis)

    except Exception as e:
        raise(e)
        
    return tuple(analyses)

'''
Start the program 
To run default config (config_path='conf', config_name='config')
To run any particular config (config_path='conf/source', config_name=['transformers', 'torch'])
'''

@hydra.main(config_path='conf', config_name='config')
def demo(cfg: DictConfig) -> None:

    # Path of Tesseract executable | MacOS Homebrew: brew install tesseract  
    pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

    print('Welcome to the model testing suite, we\'re happy to have you\n')
    logger = get_logger( 'Model Test Suite', cfg.get( 'verbose', True ))
    logger.info( f'Config:\n{OmegaConf.to_yaml(cfg)}')

    with Timer('Starting Model Test Suite', logger):

        with Timer('Creating Temporary Directory', logger):
            global temp_dir 
            temp_dir = create_temp_dir(cfg, logger)
            
        if 'create_pdf' in cfg:
            with Timer('Creating PDF', logger):
                create_pdfs(cfg, logger)
            with Timer('Converting PDF to Image', logger):
                convert_pdf(cfg, logger)

        if 'ocr' in cfg: 
            with Timer('Performing OCR', logger):
               ocr(cfg, logger)

        if 'translation_data' or 'translation_model' in cfg: 
            with Timer(f'Downloading {cfg.translation_model.model_hub} Model', logger):
                model = get_model(cfg, logger)

            with Timer('Loading Data', logger):
                data = get_data(cfg) 

            with Timer('Translating Data', logger):
                data = translate_file(data, cfg, model, logger)

        if 'translation_score' in cfg:  
            with Timer('Scoring Data', logger):
                data = calculate_scores(data, cfg, logger)
        if 'sentiment_analysis' in cfg: 
            with Timer('Analyzing Sentiment', logger):
                data = analyze_sentiment(data, cfg, logger)

        # if cfg.verbose == False: 
        #     os.remove(cfg.temp_dir)

        if 'ocr' in cfg: 
            data.to_pickle(get_original_cwd() + f'/datasets/outputs/{cfg.ocr.name}_df.pkl')
        else:
            # data.to_pickle(get_original_cwd() + f'/datasets/outputs/{cfg.translation_data.name}_df.pkl')
            data.to_pickle(get_original_cwd() + f'/datasets/outputs/test_df.pkl')

if __name__ == "__main__":
    demo()
    

