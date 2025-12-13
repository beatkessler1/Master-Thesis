import os
import openai
import logging
import dataiku
import resource
import pandas as pd

from time import time
from utils_prompt import prepare_excel_for_pdf, convert_office_to_pdf, extract_image_pdf

# Set OpenAI parameters
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

client = dataiku.api_client()
auth_info = client.get_auth_info(with_secrets=True)

for secret in auth_info['secrets']:
    if secret['key'] == 'ondrej-gpt-api-test':
        API_KEY = secret['value']
        break
        
openai.api_type = 'azure'
openai.azure_endpoint = 'https://iapi-test.merck.com/gpt/libsupport'
openai.api_version = '2025-03-01-preview'
openai.api_key = API_KEY

model_name = 'gpt-4o-2024-11-20'
price_per_mil_token = (2.5, 10)


# Read recipe inputs
folder_pref = '/tmp/auto_soe'

environ = os.environ.copy()
environ['XDG_RUNTIME_DIR'] = folder_pref

os.makedirs(folder_pref, exist_ok=True)

folder = dataiku.Folder('DYwo9FzC')
all_paths = folder.list_paths_in_partition()

output_data = dataiku.Dataset("Enterprise_Solutions_chunks")
# output_df = output_data.get_dataframe(infer_with_pandas=False)
output_df = pd.DataFrame()

log_data = dataiku.Dataset("Enterprise_Solutions_logs")
log_cols = ['directory', 'file_name', 'file_size', 'memory_total', 'page_count', 'token_count', 'extract_duration', 'price_total', 'error']

log_df = pd.DataFrame(columns=log_cols)
log_data.write_with_schema(log_df)


## Perform the extraction
price_total = 0
tokens_total = 0
output_list = []


for path in all_paths:
    basename = os.path.basename(path)
    dirname = os.path.dirname(path)

    # if dirname in output_df['Dir_Name'].values and basename in output_df['File_Name'].values:
    #     continue
    
    output_row = {
        'Dir_Name': dirname,
        'File_Name': basename,
        'Chunk_Id': 0,
        'Chunk_Text': '',
        'Chunk_Length': 0,
        'Token_Count': 0
    }

    file_size = folder.get_path_details(path)['size'] / (1024 * 1024)
    memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    
    log_row = {'directory': dirname,
               'file_name': basename,
               'file_size':  f'{file_size:.2f} MB', 
               'memory_total': f'{memory_usage:.2f} MB',
               'page_count': '', 
               'token_count': '', 
               'extract_duration': '',
               'price_total': '',
               'error': ''}

    log_df = pd.concat([log_df, pd.DataFrame([log_row])], ignore_index=True)
    log_data.write_with_schema(log_df)
    
    local_path = os.path.join(folder_pref, basename)
    _, extension = os.path.splitext(path)

    with folder.get_download_stream(path) as f_in:
        with open(local_path, 'wb') as f_out:
            f_out.write(f_in.read())
    
    if extension == '.pdf':   
        pdf_path = local_path

    elif extension in ['.pptx', '.xlsx', '.docx']:
        pdf_path = local_path.replace(extension, '.pdf')

        try:
            if extension == '.xlsx':
                if file_size > 20:
                    log_df.at[log_df.index[-1], 'error'] = 'File is too large'
                    log_data.write_with_schema(log_df)

                    output_list.append(output_row)
                    continue
                
                prepare_excel_for_pdf(local_path)
            
            convert_office_to_pdf(local_path, pdf_path, folder_pref, environ)
            print(f"File {local_path} successfully converted to pdf")

        except Exception as e:
            message = str(e)
            print(f"Error during file conversion: file: {local_path}, error: {message}")
            
            log_df.at[log_df.index[-1], 'error'] = f"File conversion issue: '{message}'"
            log_data.write_with_schema(log_df)
            
            output_list.append(output_row)
            continue
    else:
        print(f"Skipping unsupported file type: {path}")
        
        log_df.at[log_df.index[-1], 'error'] = f"Unsupported file type: '{path}'"
        log_data.write_with_schema(log_df)
        
        output_list.append(output_row)
        continue
        
    try:
        start_time = time()
        chunks, pages, tokens, price = extract_image_pdf(pdf_path, dirname, basename, openai, model_name, price_per_mil_token)
        duration = round(time() - start_time)
        
        os.remove(pdf_path)
        print(f"The content from the file {pdf_path} successfully extracted.")
            
    except Exception as e:
        message = str(e)
        print(f"Error during llm request for the file {pdf_path}: {message}")

        log_df.at[log_df.index[-1], 'error'] = f"LLM request issue: '{message}'"
        log_data.write_with_schema(log_df)
        
        output_list.append(output_row)
        continue
    
    output_list.extend(chunks)
    price_total += price
    
    print(f"Tokens: {tokens}, Duration: {duration}s, Price: {price_total:.2f}$")
    
    log_df.loc[log_df.index[-1], ['page_count', 'token_count', 'extract_duration', 'price_total']] = [f'{pages}', f'{tokens}', f'{duration}s', f'{price_total:.2f}$']
    log_data.write_with_schema(log_df)
    
    if os.path.exists(local_path):
        os.remove(local_path)


# Write recipe outputs
output_df = pd.concat([output_df, pd.DataFrame(output_list)], ignore_index=True)
output_data.write_with_schema(output_df)
