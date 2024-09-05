import os
import shutil

# Specify the directory where your text files are located
directory = './dataset/action_data'

# Iterate through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        # Remove the '.txt' extension to get the folder name
        folder_name = filename[:-4]

        # Create a new folder in the same directory with the name of the text file
        folder_path = os.path.join(directory, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # Move the text file to the newly created folder
        source = os.path.join(directory, filename)
        destination = os.path.join(folder_path, filename)
        shutil.move(source, destination)

print("Text files have been moved to their corresponding folders.")
