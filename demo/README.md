**Installing System Packages**
- **tesseract:** Required for using pytesseract
    - *MacOS | Homebrew*: 
        - brew install tesseract
        - brew install tesseract-lang

**Installing Python Packages**
- **requirements.txt**
    1. Create virtual environment 
        - venv: "python -m venv <env_name>"
        - conda: "conda create --name <env_name>"
    2. Activate virtual environment 
        - venv: " . <env_name>/bin/activate"
        - conda: "conda activate"
    3. Install requirements.txt
        - "pip install -r requirements.txt"
- **Issues with fastBPE, hydra-core**
     - Packages may need to be installed manually after running requirements.txt    
***

### evaluate.py
evaluate.py ***performs OCR*** a directory of .png's, ***translates*** the resulting OCR output using WMT19RU-EN,  caclulates ***sentiment analysis*** using pysentimiento, and outputs the data to a ***pandas dataframe(.pkl)*** to the ***/evaluate directory***

To perform the operation provided by script, follow these steps: 

1. Ensure .png files are in the /evaluate directory
2. Run "python evaluate.py absolute/path/to/evaluate"

***
### test.py
test.py enables the ***development and testing*** of the operations involving PDF Generation OCR, Translation, Translation Scoring, and Sentiment Analysis"

To enable the script, simply configure configurations files in config.yaml and the /conf directory and run "test.py"

The logging of runs will be located in the /outputs directory

***

**/datasets**
- NewsCommentary_en-ru.txt: Contains ru_en document and sentence aligned translations for NewsCommentary v16
- splitEngTxt_goodDelim: (n=1630) English Lines from the WMT22 News Dataset split into 170.txts
- splitRusTxt_goodDelim: (n=1630) Russian Lines from the WMT22 News Dataset split into 170.txts
- Tweets: (n=100) samples from the RuSentiTweet Twitter Dataset

**/conf**

The section contains the configuration directory pertaining to the Hydra configuration framework that is used in test.py. The testing framework is supported by Hydra 1.0.7: (https://hydra.cc/docs/1.0/intro/ | https://github.com/facebookresearch/hydra/releases)

The /conf directory is composed from cofiguration as such: 
- **create_pdf** (Default: OFF): Creating pdf using *reportlab*
- **ocr** (Default: OFF): Running OCR on a specified dataset/directory
- **translation** (Default: ON): Translating data, model, and scoring
    - ***translation_data***: Data used for translation
    - ***translation_model***: Model used for translation
    - ***translation_score***: Scoring used for translation
- **sentiment_analysis** (Defualt: ON): Sentiment analysis engine for sentiment analysis

**/temp**

A directory used for intermittent operations from test.py