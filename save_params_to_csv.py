def save_params_to_csv(args, training_time, strategy_type, total_parameters, start_time, end_time):
    
    defaults = {'lr': 0.001, 'epochs': 10, 'batch_size': 16, 'steps': 100}
    
    params = {
        'Learning Rate': [args.lr if args.lr is not None else defaults['lr']],
        'Epochs': [args.epochs if args.epochs is not None else defaults['epochs']],
        'Batch Size': [args.batch_size if args.batch_size is not None else defaults['batch_size']],
        'Steps per Epoch': [args.steps if args.steps is not None else defaults['steps']],
        'Start time': [start_time],
        'End time': [end_time],
        'Training Time (seconds)': [training_time],
        'Distribute Strategy': [strategy_type],
        'Model Paramters': [total_parameters]
    }
   
    df_new = pd.DataFrame(params)
    
    # Initialize Google Cloud Storage client
    client = storage.Client()
    bucket_name = args.param_file.split('/')[2]
    blob_path = '/'.join(args.param_file.split('/')[3:])
    print("Bucket Name: ", bucket_name)
    print("Blob Path: ", blob_path) #File Path 
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    
    # Try to load existing data and append new data
    try:
        # Download the existing file to a buffer
        print("Download the existing file if exists...")
        buffer = blob.download_as_text()
        df_existing = pd.read_csv(StringIO(buffer))
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
        print("Appended the new data in the existing file...")
    except Exception as e:
        print(f"Error downloading the file: {e}. A new file will be created.")
        # If the file does not exist or other errors occur, use new data
        print("Creating new file ...")
        df_final = df_new
    
    print("Output param file location: ", args.param_file)
    
    try:
        # Convert DataFrame to CSV and upload back to the GCS
        print("Will upload the file in GCS ....")
        blob.reload()
        blob.upload_from_string(df_final.to_csv(index=False), 'text/csv')
        
        # df.to_csv(args.param_file, index=False)
        print("Param file saved....")
    except google.api_core.exceptions.BadRequest as e:
        print(f"Failed to upload due to a bad request: {e.message}")
        if hasattr(e, 'response') and e.response:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response content: {e.response.content}")
