# IS_VIDDL

Add the text classification python file in the Google Cloud Vertex AI platform. Create a training job and run the job.

Once the training has been completed, collect the csv file for each parameters from GCS(Google CLoud Storage) --> Monitoring dashboard and place it in Data folder.

Then run the script combine_data.py to combine all the csv files into one and save it as 'combined_data.csv' in Data folder.

ALso download the param.csv from the GCS bucket.
