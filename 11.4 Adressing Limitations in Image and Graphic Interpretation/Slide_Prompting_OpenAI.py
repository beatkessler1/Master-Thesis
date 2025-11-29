{
  "metadata": {
    "kernelspec": {
      "name": "py-dku-containerized-venv-genai-bootcamp-env-general_4gb_1cpu",
      "display_name": "Python in general_4GB_1CPU (env genai-bootcamp-env)",
      "language": "python"
    },
    "hide_input": false,
    "language_info": {
      "name": "python",
      "version": "3.9.20",
      "mimetype": "text/x-python",
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "pygments_lexer": "ipython3",
      "nbconvert_exporter": "python",
      "file_extension": ".py"
    },
    "customFields": {},
    "createdOn": 1726542164366,
    "modifiedBy": "kesslebe",
    "tags": [],
    "creator": "ondrej"
  },
  "nbformat": 4,
  "nbformat_minor": 1,
  "cells": [
    {
      "execution_count": 0,
      "cell_type": "code",
      "metadata": {},
      "source": [
        "import io\nimport os\nimport fitz\nimport pypdf\nimport openai\nimport base64\nimport dataiku\nimport tiktoken\nimport subprocess\nimport pandas as pd\n\nfrom time import time\nfrom PIL import Image\nfrom IPython.display import display, Markdown\n\n\ndef convert_pptx_to_pdf(input_file, output_file):\n    # Using LibreOffice to convert pptx to pdf via command line\n    # LibreOffice headless mode is used for running without GUI\n    try:\n        output_dir \u003d os.path.dirname(output_file)\n        \n        env \u003d os.environ.copy()\n        env[\u0027XDG_RUNTIME_DIR\u0027] \u003d folder_pref\n        \n        command \u003d [\u0027libreoffice\u0027, \u0027--headless\u0027, \u0027--convert-to\u0027, \u0027pdf\u0027, input_file, \u0027--outdir\u0027, output_dir]\n        subprocess.run(command, env\u003denv, check\u003dTrue)\n        \n        print(f\"Conversion successful: {output_file}\")\n        \n    except subprocess.CalledProcessError as e:\n        print(f\"Error during conversion: {e}\")\n\n\ndef pdf_to_base64(pdf_path, page_num, zoom\u003d1.5):\n    document \u003d fitz.open(pdf_path)\n    page \u003d document.load_page(page_num)\n\n    # Convert PDF page to a pixmap (image)\n    pix \u003d page.get_pixmap(matrix\u003dfitz.Matrix(zoom, zoom))\n\n    # Create an Image object from the pixmap (image)\n    image \u003d Image.frombytes(\u0027RGB\u0027, [pix.width, pix.height], pix.samples)\n\n    # Save the image to a bytes buffer\n    buffered \u003d io.BytesIO()\n    image.save(buffered, format\u003d\u0027JPEG\u0027)\n\n    # Encode the image bytes to a base64 string\n    base64_string \u003d base64.b64encode(buffered.getvalue()).decode(\u0027utf-8\u0027)\n\n    # Format the result with the appropriate prefix\n    base64_string \u003d f\u0027data:image/jpeg;base64,{base64_string}\u0027\n    document.close()\n    \n    return base64_string, image\n\nprint(\"test\")"
      ],
      "outputs": []
    },
    {
      "execution_count": 0,
      "cell_type": "code",
      "metadata": {},
      "source": [
        "client \u003d dataiku.api_client()\nauth_info \u003d client.get_auth_info(with_secrets\u003dTrue)\n\nfor secret in auth_info[\u0027secrets\u0027]:\n    if secret[\u0027key\u0027] \u003d\u003d \u0027neura-gpt-test-key\u0027:\n        API_KEY \u003d secret[\u0027value\u0027]\n        break\n        \nopenai.api_type \u003d \u0027azure\u0027\nopenai.azure_endpoint \u003d \u0027https://iapi-test.merck.com/gpt/libsupport\u0027\nopenai.api_version \u003d \u00272024-12-01-preview\u0027\nopenai.api_key \u003d API_KEY"
      ],
      "outputs": []
    },
    {
      "execution_count": 0,
      "cell_type": "code",
      "metadata": {},
      "source": [
        "# Read recipe inputs\nfolder_id \u003d \u0027kEfrpMeV\u0027 \nfolder_pref \u003d \u0027/tmp/auto_soe\u0027\n\nos.makedirs(folder_pref, exist_ok\u003dTrue)\n\n\n# Read recipe inputs\nfolder \u003d dataiku.Folder(folder_id)\nall_paths \u003d folder.list_paths_in_partition()\n# path \u003d all_paths[0]\n# path \u003d \u0027/Supply Chain Stage Code.pdf\u0027\npath \u003d \u0027/Doravirine Backup Document (6).pdf\u0027\n\nlocal_path \u003d folder_pref + path\n\nwith folder.get_download_stream(path) as f_in:\n    with open(local_path, \u0027wb\u0027) as f_out:\n        f_out.write(f_in.read())\n\nprint(path)\nprint(local_path)"
      ],
      "outputs": []
    },
    {
      "execution_count": 0,
      "cell_type": "code",
      "metadata": {},
      "source": [
        "if local_path.endswith(\u0027.pdf\u0027):\n    pdf_path \u003d local_path\n    \nelif local_path.endswith(\u0027.pptx\u0027):\n    pdf_path \u003d local_path.replace(\u0027.pptx\u0027, \u0027.pdf\u0027)\n    convert_pptx_to_pdf(local_path, pdf_path)\nelse:\n    print(f\u0027Unsuported file type: {local_path}\u0027)\n\nprint(pdf_path)"
      ],
      "outputs": []
    },
    {
      "execution_count": 0,
      "cell_type": "code",
      "metadata": {},
      "source": [
        "image_data \u003d []\npage_numbers \u003d [5]\n\nfor page_num in page_numbers:\n    base64_string, image \u003d pdf_to_base64(pdf_path, page_num)\n    display(image)\n    \n    image_item \u003d {\n        \u0027type\u0027: \u0027image_url\u0027,\n        \u0027image_url\u0027: {\n            \u0027url\u0027: base64_string,\n        }\n    }\n    \n    image_data.append(image_item)\n    \nimport pypdf\nfrom io import BytesIO\n\n\n# Open the PDF file\nwith open(pdf_path, \u0027rb\u0027) as file:\n    reader \u003d pypdf.PdfReader(BytesIO(file.read()))\n        \n    texts \u003d []\n    \n    for page_num in page_numbers:\n        page \u003d reader.pages[page_num]\n        texts.append(page.extract_text())\n\nMarkdown(texts[0].replace(\u0027\\n\u0027, \u0027  \\n\u0027))"
      ],
      "outputs": []
    },
    {
      "execution_count": 0,
      "cell_type": "code",
      "metadata": {},
      "source": [
        "import pypdf\nfrom io import BytesIO\n\n\n# Open the PDF file\nwith open(pdf_path, \u0027rb\u0027) as file:\n    reader \u003d pypdf.PdfReader(BytesIO(file.read()))\n        \n    texts \u003d []\n    \n    for page_num in page_numbers:\n        page \u003d reader.pages[page_num]\n        texts.append(page.extract_text())\n\nMarkdown(texts[0].replace(\u0027\\n\u0027, \u0027  \\n\u0027))"
      ],
      "outputs": []
    },
    {
      "execution_count": 0,
      "cell_type": "code",
      "metadata": {},
      "source": [
        "# prompt \u003d f\u0027\u0027\u0027You are given a slide from a PowerPoint presentation. Your task is to extract all relevant information from it.\n# It is important to keep the extract accurate, comprehensive and also well structured, as the extracted information will be later used as an input to a RAG model.\n# \u0027\u0027\u0027\n\n# prompt \u003d f\u0027\u0027\u0027You are given a slide from a PowerPoint presentation. Your task is to extract all relevant information from it.\n# It is important to keep the extract accurate, comprehensive and also well structured, as the extracted information will be later used as an input to a RAG model.\n# The information on the slide is mainly in the form of one of these:\n# - chart: if you identify a chart, provide a detailed description of its content and also the summary of what the chart depicts\n# - table: if there is a table present, extract it with the original table formatting to preserve the hierarchical structure\n# - text: if the object is neither chart nor table, extract the remaining text as it is while keeping the original text formatting\n# At the end, provide a summarized description of the slide as a whole.\n# \u0027\u0027\u0027\n\nprompt \u003d f\u0027\u0027\u0027\nYou are given a slide from a PowerPoint presentation. Your task is to extract all relevant information from it. It is important to keep the extract accurate, comprehensive, and well-structured, as the extracted information will be used as input for a Retrieval-Augmented Generation (RAG) model.\n\nPlease follow these detailed instructions:\n\n1. **Identify and Extract Content**:\n   - **Chart**: If you identify a chart, provide a detailed description of its content. Include:\n     - The type of chart (e.g., bar, line, pie, etc.).\n     - The axes labels and their units.\n     - The data points or trends observed.\n     - A summary of what the chart depicts.\n   \n   - **Table**: If there is a table present, extract it while preserving the original table formatting. Ensure:\n     - All headers and sub-headers are correctly identified.\n     - The hierarchical structure of the table is maintained.\n     - Any notable data points or trends are highlighted.\n   \n   - **Text**: If the object is neither a chart nor a table, extract the remaining text as it is while keeping the original text formatting. Ensure:\n     - All bullet points, numbering, and indentations are preserved.\n     - Any emphasized text (bold, italics, etc.) is accurately represented.\n     \n2. **Summarize the Slide**:\n   - Provide a summarized description of the slide, capturing the main points and purpose of the slide.\n\u0027\u0027\u0027\n\ncontent \u003d [{\u0027type\u0027: \u0027text\u0027, \u0027text\u0027: prompt}] + image_data\n\nprint(\"test\")"
      ],
      "outputs": []
    },
    {
      "execution_count": 0,
      "cell_type": "code",
      "metadata": {},
      "source": [
        "start_time \u003d time()\n\n# Prepare parameters based on model\nmodel_name \u003d \u0027gpt-5-2025-08-07\u0027  # Removed trailing space  gpt-5-2025-08-07 gpt-4o-2024-05-13\nparams \u003d {\n    \u0027model\u0027: model_name,\n    \u0027messages\u0027: [\n        {\u0027role\u0027: \u0027user\u0027, \u0027content\u0027: \u0027Formatting re-enabled\u0027},  # ✅ Direct string like your working example\n        {\u0027role\u0027: \u0027user\u0027, \u0027content\u0027: content}  # Make sure \u0027content\u0027 variable exists\n    ]\n}\n\n# GPT-5 specific parameters\nif \u0027gpt-5\u0027 in model_name.lower():\n    params[\u0027reasoning_effort\u0027] \u003d \u0027minimal\u0027\n    params[\u0027verbosity\u0027] \u003d \u0027low\u0027\n    params[\u0027messages\u0027][0][\u0027role\u0027] \u003d \u0027developer\u0027  # Use \u0027developer\u0027 role for GPT-5\n\nresponse \u003d openai.chat.completions.create(**params)\n\nprint(f\u0027Price: {(response.usage.prompt_tokens + 3 * response.usage.completion_tokens) / 1000000 * 5:.2f}$\u0027)\nprint(f\u0027Tokens: {response.usage.total_tokens}\u0027)\nprint(f\u0027Duration: {round(time() - start_time)}s\u0027)\n"
      ],
      "outputs": []
    },
    {
      "execution_count": 0,
      "cell_type": "code",
      "metadata": {},
      "source": [
        "display(Markdown(response.choices[0].message.content))"
      ],
      "outputs": []
    },
    {
      "execution_count": 0,
      "cell_type": "code",
      "metadata": {},
      "source": [
        "display(Markdown(response.choices[0].message.content))"
      ],
      "outputs": []
    },
    {
      "execution_count": 0,
      "cell_type": "code",
      "metadata": {},
      "source": [
        "os.remove(local_path)\nos.remove(pdf_path)\nprint(os.listdir(folder_pref))"
      ],
      "outputs": []
    },
    {
      "execution_count": 0,
      "cell_type": "code",
      "metadata": {},
      "source": [
        "encoder \u003d tiktoken.encoding_for_model(\u0027gpt-4\u0027)\n\nprint(f\u0027Tokens_completion: {response.usage.completion_tokens}\u0027)\nprint(f\u0027Tokens_encoder: {len(encoder.encode(response.choices[0].message.content))}\u0027)"
      ],
      "outputs": []
    },
    {
      "execution_count": 0,
      "cell_type": "code",
      "metadata": {},
      "source": [
        ""
      ],
      "outputs": []
    },
    {
      "execution_count": 0,
      "cell_type": "code",
      "metadata": {},
      "source": [
        ""
      ],
      "outputs": []
    },
    {
      "execution_count": 0,
      "cell_type": "code",
      "metadata": {},
      "source": [
        ""
      ],
      "outputs": []
    }
  ]
}
