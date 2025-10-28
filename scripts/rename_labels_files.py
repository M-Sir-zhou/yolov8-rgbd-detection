import os

def rename_labels_files(folder_path):
    for filename in os.listdir(folder_path):
        if "labels" in filename:
            new_filename = filename.replace("labels", "")
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")

if __name__ == "__main__":
    folder_path = "D:\\ProjectCode\\PyCharm\\ultralytics-main\\datasets\\tennis_dark\\val\\labels"
    rename_labels_files(folder_path)