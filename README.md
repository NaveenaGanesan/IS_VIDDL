# IS_VIDDL

## Run the models on Vertex AI and collect data

1. Go to the GCP(Google Cloud Platform): https://console.cloud.google.com/
2. Search for Vertex AI and create a new instance in the workbench under Notebooks
3. From the created instance, open the JupyterLab and add the text_classification.ipynb file. Then run the file
4. This python file will have the model code and the required setup to run the model. It created a GCS(Google CLoud Storage) bucket and adds the compressed model package.
5. It also creates a training job and run the model from the GCS.
6. Once the training has been completed, collect the csv file for each parameters from GCS(Google CLoud Storage) --> Monitoring dashboard and place it in Data folder.
7. Then run the script combine_data.py to combine all the csv files into one and save it as 'combined_data.csv' in Data folder.
8. Also to obtain the model parameters, download the param.csv from the GCS bucket. This file has the model parameters collected during the training
