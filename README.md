# sta4930-capstone
 Repository for STA4930 Project

### Objective Breakdown for Thursday, February 9th
*** 
**Set up testing framework for our OCR and Translation models** 
- OCR
	- Testing Framework: Online Article -> Format for OCR (i.e. pdf/png) -> OCR -> Metrics -> .csv/.txt
		- Online Article
			- Data Collection: Find around several articles on different subjects
		- Format for OCR
			- Convert article url into a downloadable pdf (probably a package for this)
		- OCR  
			- Run OCR using tesseract (python wrapper?)
		- Metrics
			- Establish accuracy metrics
		- Optional: 
			- .csv/.txt
				- Convert in such a way that 1 article = 1 file
- Translation 
	- Testing Framework:  .csv/.txt -> Pre-processing -> Translate -> Metrics
		- .csv/.txt
			- Setup directory to files and ingest
		- Pre-processing
			- Lines -> (i.e.  words, sentences, paragraphs) for translation   
		- Translate
			- Run Translation for each article 
		- Metrics
			- Publish BLEU/METEOR Scores
- Data Analytics (OCR Metrics/Translation Metrics)
			- Dashboard/Plots/etc.
   
   
### Model Git Directory
***
	- project directory/
		- testing/
			- ocr/
				- .gitignore
				- requirements.txt or conda equivalent
				- data/
					- image_1.csv
					- image_2.csv
					- image_3.csv
			- translation/
				- models/
					- model 1/
						- python_version.txt
						- .gitignore
						- requirements.txt or conda equivalent
					- model 2/
      - python_version.txt
						- .gitignore
						- requirements.txt or conda equivalent
					- model 3/
      - python_version.txt
						- .gitignore
						- requirements.txt or conda equivalent


### Instructions for branching, pulling, pushing, directory-setup 
***
- **Branch "ocr-test-framework"** <-|-> **Only push to ocr-test-framework**
	- Handle .gitignore as you like
	- Ensure you have requirements.txt
	- Partition a folder for data to be accessed by testing framework
- **Branch "tran-test-framework"** <-|-> **Only push to tran-test-framework**
	- Only edit the model you have been assigned 
	- Each model should persist in a folder under models/ with the folder named after the model as such:
	- Example: Models/Joey_NMT/everything related to Joey_NMT
	- models/
		- model1/
			- .gitignore
				- Include env or conda equivalent. These folders manage your virtual environment and contain your packages. Instead use requirements.txt
				- Ignore any other local files you wouldn't want to upload to Github. 
					- This will typically any large data sets or models. Github has a file size limit of 20MB
			- requirements.txt
				- If using python venv
					- pip freeze > requirements.txt
   - python_version.txt
    - Should be a .txt file with just the python version (i.e. 3.9.13)

### Extra comments
***
**On generating requirements.txt**
1.  Manually: You can manually create a `requirements.txt` file and list the dependencies and their versions that your project requires.
2.  Using pip freeze: If you already have your dependencies installed in your virtual environment, you can use the `pip freeze` command to generate a list of the packages and their versions, which you can then save to a `requirements.txt` file.
3.  Using setup tools: You can use the `setup` module from the `setuptools` library to specify the dependencies in your `setup.py` file and then generate a `requirements.txt` file using the command `pip freeze > requirements.txt`.
4.  Using pipenv: If you're using `pipenv` to manage your dependencies, you can generate a `requirements.txt` file using the `pipenv lock` command, which will lock the versions of the dependencies and generate a `Pipfile.lock` file. You can then use the `pipenv lock -r` command to generate a `requirements.txt` file from the `Pipfile.lock` file.

**Naming Conventions**
1.  Modules: Modules should have short, all-lowercase names, and they should use the underscore character to separate words, if needed (e.g., `my_module.py`).
2.  Classes: Class names should be written in CamelCase (e.g., `MyClass`).
3.  Variables: Variable names should be lowercase, with words separated by underscores (e.g., `my_variable`).
4.  Functions: Function names should be lowercase, with words separated by underscores (e.g., `my_function()`).
5.  Constants: Constants should be written in all uppercase, with words separated by underscores (e.g., `MY_CONSTANT`).
