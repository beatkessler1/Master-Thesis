import io
import os
import fitz
import pypdf
import openai
import base64
import dataiku
import tiktoken
import subprocess
import pandas as pd

from time import time
from PIL import Image
from IPython.display import display, Markdown


def convert_pptx_to_pdf(input_file, output_file):
    # Using LibreOffice to convert pptx to pdf via command line
    # LibreOffice headless mode is used for running without GUI
    try:
        output_dir = os.path.dirname(output_file)
        
        env = os.environ.copy()
        env['XDG_RUNTIME_DIR'] = folder_pref
        
        command = ['libreoffice', '--headless', '--convert-to', 'pdf', input_file, '--outdir', output_dir]
        subprocess.run(command, env=env, check=True)
        
        print(f"Conversion successful: {output_file}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")


def pdf_to_base64(pdf_path, page_num, zoom=1.5):
    document = fitz.open(pdf_path)
    page = document.load_page(page_num)

    # Convert PDF page to a pixmap (image)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))

    # Create an Image object from the pixmap (image)
    image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)

    # Save the image to a bytes buffer
    buffered = io.BytesIO()
    image.save(buffered, format='JPEG')

    # Encode the image bytes to a base64 string
    base64_string = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Format the result with the appropriate prefix
    base64_string = f'data:image/jpeg;base64,{base64_string}'
    document.close()
    
    return base64_string, image

print("test")


client = dataiku.api_client()
auth_info = client.get_auth_info(with_secrets=True)

for secret in auth_info['secrets']:
    if secret['key'] == 'neura-gpt-test-key':
        API_KEY = secret['value']
        break
        
openai.api_type = 'azure'
openai.azure_endpoint = 'https://iapi-test.merck.com/gpt/libsupport'
openai.api_version = '2024-12-01-preview'
openai.api_key = API_KEY
# Read recipe inputs
folder_id = 'kEfrpMeV' 
folder_pref = '/tmp/auto_soe'

os.makedirs(folder_pref, exist_ok=True)


# Read recipe inputs
folder = dataiku.Folder(folder_id)
all_paths = folder.list_paths_in_partition()
# path = all_paths[0]
# path = '/Supply Chain Stage Code.pdf'
path = '/Doravirine Backup Document (6).pdf'

local_path = folder_pref + path

with folder.get_download_stream(path) as f_in:
    with open(local_path, 'wb') as f_out:
        f_out.write(f_in.read())

print(path)
print(local_path)
if local_path.endswith('.pdf'):
    pdf_path = local_path
    
elif local_path.endswith('.pptx'):
    pdf_path = local_path.replace('.pptx', '.pdf')
    convert_pptx_to_pdf(local_path, pdf_path)
else:
    print(f'Unsuported file type: {local_path}')

print(pdf_path)

image_data = []
page_numbers = [5]

for page_num in page_numbers:
    base64_string, image = pdf_to_base64(pdf_path, page_num)
    display(image)
    
    image_item = {
        'type': 'image_url',
        'image_url': {
            'url': base64_string,
        }
    }
    
    image_data.append(image_item)
    
import pypdf
from io import BytesIO


# Open the PDF file
with open(pdf_path, 'rb') as file:
    reader = pypdf.PdfReader(BytesIO(file.read()))
        
    texts = []
    
    for page_num in page_numbers:
        page = reader.pages[page_num]
        texts.append(page.extract_text())

Markdown(texts[0].replace('\n', '  \n'))
import pypdf
from io import BytesIO


# Open the PDF file
with open(pdf_path, 'rb') as file:
    reader = pypdf.PdfReader(BytesIO(file.read()))
        
    texts = []
    
    for page_num in page_numbers:
        page = reader.pages[page_num]
        texts.append(page.extract_text())

Markdown(texts[0].replace('\n', '  \n'))
# prompt = f'''You are given a slide from a PowerPoint presentation. Your task is to extract all relevant information from it.
# It is important to keep the extract accurate, comprehensive and also well structured, as the extracted information will be later used as an input to a RAG model.
# '''

# prompt = f'''You are given a slide from a PowerPoint presentation. Your task is to extract all relevant information from it.
# It is important to keep the extract accurate, comprehensive and also well structured, as the extracted information will be later used as an input to a RAG model.
# The information on the slide is mainly in the form of one of these:
# - chart: if you identify a chart, provide a detailed description of its content and also the summary of what the chart depicts
# - table: if there is a table present, extract it with the original table formatting to preserve the hierarchical structure
# - text: if the object is neither chart nor table, extract the remaining text as it is while keeping the original text formatting
# At the end, provide a summarized description of the slide as a whole.
# '''

prompt = f'''
You are given a slide from a PowerPoint presentation. Your task is to extract all relevant information from it. It is important to keep the extract accurate, comprehensive, and well-structured, as the extracted information will be used as input for a Retrieval-Augmented Generation (RAG) model.

Please follow these detailed instructions:

1. **Identify and Extract Content**:
   - **Chart**: If you identify a chart, provide a detailed description of its content. Include:
     - The type of chart (e.g., bar, line, pie, etc.).
     - The axes labels and their units.
     - The data points or trends observed.
     - A summary of what the chart depicts.
   
   - **Table**: If there is a table present, extract it while preserving the original table formatting. Ensure:
     - All headers and sub-headers are correctly identified.
     - The hierarchical structure of the table is maintained.
     - Any notable data points or trends are highlighted.
   
   - **Text**: If the object is neither a chart nor a table, extract the remaining text as it is while keeping the original text formatting. Ensure:
     - All bullet points, numbering, and indentations are preserved.
     - Any emphasized text (bold, italics, etc.) is accurately represented.
     
2. **Summarize the Slide**:
   - Provide a summarized description of the slide, capturing the main points and purpose of the slide.
'''

content = [{'type': 'text', 'text': prompt}] + image_data

print("test")
start_time = time()

model_name = 'gpt-5-2025-08-07'  # gpt-5-2025-08-07 gpt-4o-2024-05-13
params = {
    'model': model_name,
    'messages': [
        {'role': 'user', 'content': 'Formatting re-enabled'},  
        {'role': 'user', 'content': content}  
    ]
}
# GPT-5 specific parameters
if 'gpt-5' in model_name.lower():
    params['reasoning_effort'] = 'minimal'
    params['verbosity'] = 'low'
    params['messages'][0]['role'] = 'developer'  # Use 'developer' role for GPT-5

response = openai.chat.completions.create(**params)

print(f'Price: {(response.usage.prompt_tokens + 3 * response.usage.completion_tokens) / 1000000 * 5:.2f}$')
print(f'Tokens: {response.usage.total_tokens}')
print(f'Duration: {round(time() - start_time)}s'
