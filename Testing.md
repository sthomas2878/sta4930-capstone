# Notes on Configurations for Experiements
***

The script is based on the Hydra Framework, a framework developed by Facebook for configuring and running code using different parameters. 

There is one main differentiating factor involved with the configuration of this script, which is the model framework from which the model is derived. The key functions peratining to the model framework configuration include: download, processing (tokenization), translation. The two model frameworks used are: 
    - Hugging Face Transformers
    - Torch Hub

To ensure that model configurations are constant across model frameworks configurations, model configurations 
