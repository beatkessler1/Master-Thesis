import os
import openai
import logging
import dataiku
import resource
import pandas as pd
import threading
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time
from utils_prompt import prepare_excel_for_pdf, convert_office_to_pdf, extract_image_pdf_2

# Set OpenAI parameters
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
client = dataiku.api_client()
auth_info = client.get_auth_info(with_secrets=True)
for secret in auth_info['secrets']:
    if secret['key'] == 'neura-gpt-test-key':
        API_KEY = secret['value']
        break
        
openai.api_type = 'azure'
openai.azure_endpoint = 'https://iapi-test.merck.com/gpt/libsupport'
openai.api_version = '2025-03-01-preview'
openai.api_key = API_KEY
model_name = 'gpt-5-2025-08-07'
price_per_mil_token = (1.25, 10)

# Read recipe inputs
folder_pref = '/tmp/auto_soe'
environ = os.environ.copy()
environ['XDG_RUNTIME_DIR'] = folder_pref
os.makedirs(folder_pref, exist_ok=True)

folder = dataiku.Folder('NRmWnZPT')
all_paths = folder.list_paths_in_partition()

# Initialize datasets
output_data = dataiku.Dataset("QD332C")
log_data = dataiku.Dataset("QD332L")

# Define the schemas explicitly
output_schema = [
    {"name": "Dir_Name", "type": "string"},
    {"name": "File_Name", "type": "string"}, 
    {"name": "Chunk_Id", "type": "int"},
    {"name": "Chunk_Text", "type": "string"},
    {"name": "Chunk_Length", "type": "int"},
    {"name": "Token_Count", "type": "int"}
]

log_schema = [
    {"name": "directory", "type": "string"},
    {"name": "file_name", "type": "string"},
    {"name": "file_size", "type": "string"},
    {"name": "memory_total", "type": "string"},
    {"name": "page_count", "type": "string"},
    {"name": "token_count", "type": "string"},
    {"name": "extract_duration", "type": "string"},
    {"name": "price_total", "type": "string"},
    {"name": "error", "type": "string"}
]

# Clear datasets and set schemas
output_data.write_schema(output_schema, drop_and_create=True)
log_data.write_schema(log_schema, drop_and_create=True)

# Get writers for appending
output_writer = output_data.get_writer()
log_writer = log_data.get_writer()

# Thread-safe variables
price_total = 0
processed_count = 0
write_lock = threading.Lock()
price_lock = threading.Lock()

def clean_filename_unicode_safe(filename):
    """Clean filename while preserving Unicode characters"""
    # Normalize Unicode characters
    filename = unicodedata.normalize('NFC', filename)
    # Only replace characters that are problematic for file systems
    cleaned = re.sub(r'[<>:"/\\|?*\s]', '_', filename)  # Only filesystem-unsafe chars
    # Remove multiple consecutive underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    return cleaned

def process_file(path):
    """Process a single file and immediately write results"""
    global price_total, processed_count
    
    basename = os.path.basename(path)
    dirname = os.path.dirname(path)
    
    output_row = {
        'Dir_Name': dirname,
        'File_Name': basename,
        'Chunk_Id': 0,
        'Chunk_Text': '',
        'Chunk_Length': 0,
        'Token_Count': 0
    }
    
    try:
        file_size = folder.get_path_details(path)['size'] / (1024 * 1024)
        memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception as e:
        print("Error getting file details for {}: {}".format(path, e))
        file_size = 0
        memory_usage = 0
    
    log_row = {
        'directory': dirname,
        'file_name': basename,
        'file_size': '{:.2f} MB'.format(file_size), 
        'memory_total': '{:.2f} MB'.format(memory_usage),
        'page_count': '', 
        'token_count': '', 
        'extract_duration': '',
        'price_total': '',
        'error': ''
    }
    
    # Use shorter thread ID and Unicode-safe filename cleaning
    thread_id = threading.current_thread().ident
    short_thread_id = str(thread_id)[-6:]  # Only last 6 digits
    
    # Unicode-safe filename cleaning - preserves Chinese characters
    clean_basename = clean_filename_unicode_safe(basename)
    
    # Create shorter, cleaner filename
    local_path = os.path.join(folder_pref, "{}_{}".format(short_thread_id, clean_basename))
    
    pdf_path = None
    chunks_result = [output_row]
    price_result = 0
    
    try:
        _, extension = os.path.splitext(path)
        
        # Download file - exactly like your old code
        with folder.get_download_stream(path) as f_in:
            with open(local_path, 'wb') as f_out:
                f_out.write(f_in.read())
        
        # Process based on file type - old approach + .doc formats
        if extension == '.pdf':   
            pdf_path = local_path

        elif extension in ['.pptx', '.xlsx', '.docx', '.doc']:  # Added .doc to old working list
            pdf_path = local_path.replace(extension, '.pdf')

            try:
                if extension == '.xlsx':  # Keep original Excel handling (only modern .xlsx)
                    if file_size > 20:
                        log_row['error'] = 'File is too large'
                        with write_lock:
                            log_writer.write_row_dict(log_row)
                            output_writer.write_row_dict(output_row)
                        return
                    
                    prepare_excel_for_pdf(local_path)
                
                convert_office_to_pdf(local_path, pdf_path, folder_pref, environ)
                print("File {} successfully converted to pdf".format(local_path))

            except Exception as e:
                message = str(e)
                print("Error during file conversion: file: {}, error: {}".format(local_path, message))
                
                log_row['error'] = "File conversion issue: '{}'".format(message)
                with write_lock:
                    log_writer.write_row_dict(log_row)
                    output_writer.write_row_dict(output_row)
                return
        else:
            print("Skipping unsupported file type: {}".format(path))
            
            log_row['error'] = "Unsupported file type: '{}'".format(path)
            with write_lock:
                log_writer.write_row_dict(log_row)
                output_writer.write_row_dict(output_row)
            return
        
        # Extract content
        try:
            start_time = time()
            chunks, pages, tokens, price = extract_image_pdf_2(
                pdf_path, dirname, basename, openai, model_name, price_per_mil_token
            )
            duration = round(time() - start_time)
            
            chunks_result = chunks
            price_result = price
            
            print("Extracted {} chunks from {} ({} tokens, ${:.2f})".format(
                len(chunks), basename, tokens, price))
            
            # Update log row with success info
            log_row.update({
                'page_count': str(pages),
                'token_count': str(tokens),
                'extract_duration': '{}s'.format(duration),
                'price_total': '{:.2f}$'.format(price)
            })
                
        except Exception as e:
            message = str(e)
            print("Error during llm request for the file {}: {}".format(pdf_path, message))
            
            log_row['error'] = "LLM request issue: '{}'".format(message)
            with write_lock:
                log_writer.write_row_dict(log_row)
                output_writer.write_row_dict(output_row)
            return
        
    except Exception as e:
        log_row['error'] = "Unexpected error: {}".format(str(e))
        print("Unexpected error processing {}: {}".format(path, e))
        
    finally:
        # Clean up files
        cleanup_files = []
        if local_path and os.path.exists(local_path):
            cleanup_files.append(local_path)
        if pdf_path and pdf_path != local_path and os.path.exists(pdf_path):
            cleanup_files.append(pdf_path)
            
        for cleanup_file in cleanup_files:
            try:
                os.remove(cleanup_file)
            except Exception as e:
                print("Error cleaning up file {}: {}".format(cleanup_file, e))
    
    # Write results immediately using the writer (thread-safe)
    with write_lock:
        # Write each chunk individually using write_row_dict
        for chunk in chunks_result:
            output_writer.write_row_dict(chunk)
        
        # Write log entry
        log_writer.write_row_dict(log_row)
    
    # Update global price counter (thread-safe)
    with price_lock:
        price_total += price_result
        processed_count += 1
        
        # Progress update every 10 files
        if processed_count % 10 == 0:
            print("Progress: {}/{} files completed. Total cost: ${:.2f}".format(
                processed_count, len(all_paths), price_total))

# Process all files
total_files = len(all_paths)
print("Starting processing of {} files".format(total_files))

try:
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        future_to_path = {executor.submit(process_file, path): path for path in all_paths}
        
        # Process results as they complete
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                future.result()
            except Exception as e:
                print("Critical error processing {}: {}".format(path, e))
                # Still log the error
                with write_lock:
                    error_log = {
                        'directory': os.path.dirname(path),
                        'file_name': os.path.basename(path),
                        'file_size': '', 'memory_total': '', 'page_count': '',
                        'token_count': '', 'extract_duration': '', 'price_total': '',
                        'error': 'Critical failure: {}'.format(str(e))
                    }
                    log_writer.write_row_dict(error_log)

finally:
    # Close the writers
    output_writer.close()
    log_writer.close()

print("Total processing completed. Processed {} files. Total price: ${:.2f}".format(
    processed_count, price_total))

