import os


def rename_jpg_files(folder_path):
    # Get the folder name
    folder_name = os.path.basename(folder_path)

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a JPG
        if filename.lower().endswith('.jpg'):
            # Create the new filename
            new_filename = f"{folder_name}_{filename}"

            # Create full file paths
            old_file = os.path.join(folder_path, filename)
            new_file = os.path.join(folder_path, new_filename)

            # Rename the file
            os.rename(old_file, new_file)
            print(f"Renamed: {filename} to {new_filename}")


# Usage
folder_path = 'images_calib3'
rename_jpg_files(folder_path)

