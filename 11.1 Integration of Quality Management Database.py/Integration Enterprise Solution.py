import dataiku
from office365.runtime.auth.user_credential import UserCredential
from office365.sharepoint.client_context import ClientContext
from utils import getAuthSession, get_files_in_folder        
base_url = 'https://collaboration.merck.com'
site_path = '/sites/MMD_Ent_Sol_Docs'

username, password = getAuthSession(kerberos=False)

user_credentials = UserCredential(f'{username}@merck.com', password)
ctx = ClientContext(base_url + site_path).with_credentials(user_credentials)

output_folder = dataiku.Folder('DYwo9FzC')
output_path = output_folder.get_path()

folder_path = site_path + '/Shared Documents'

suffix_count = dict()
error_files = list()

for file_path in get_files_in_folder(ctx, folder_path):
    file_name = os.path.basename(file_path)
    relative_path = os.path.relpath(file_path, folder_path)
    
    # Avoid downloading Excel files
    _, file_extension = os.path.splitext(file_name)
    
    if file_extension not in ['.pptx', '.docx', '.pdf', '.xlsx', '.xls', '.xlsm']:
        print(f'Skipping file {relative_path} as it is not of the required type.')
        continue
    
    # Create the full directory path locally
    local_dir = os.path.dirname(relative_path)
    local_dir_full = os.path.join(output_path, local_dir)
    
    os.makedirs(local_dir_full, exist_ok=True)
    
    try:
        # Download the file to a local file path
        file = ctx.web.get_file_by_server_relative_url(file_path)
        ctx.load(file).execute_query()

        local_file_path = os.path.join(local_dir_full, file_name)

        with open(local_file_path, 'wb') as local_file:
            file.download(local_file).execute_query()

        print(f'File {relative_path} downloaded successfully.')
        
        if file_extension in suffix_count.keys():
            suffix_count[file_extension] += 1
        else:
            suffix_count[file_extension] = 0
        
    except Exception as e:
        error_files.append(relative_path)
        print(f'Failed to download file {relative_path}: {e}')

print(f'Suffix Count: {suffix_count}')
print(f'Error Count: {error_files}')
