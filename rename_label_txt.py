import os

# Specify the directory where your text files are located
directory = './dataset/action_data'

# Iterate through all folders in the directory
for folder_name in os.listdir(directory):
    folder_path = os.path.join(directory, folder_name)

    # Check if it's a directory
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                # Define the new file name as 'label.txt'
                new_filename = 'label.txt'
                old_file_path = os.path.join(folder_path, filename)
                new_file_path = os.path.join(folder_path, new_filename)

                # Rename the text file to 'label.txt'
                os.rename(old_file_path, new_file_path)

print("Text files have been renamed to 'label.txt' in their respective folders.")
