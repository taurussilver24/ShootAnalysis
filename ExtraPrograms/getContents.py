import os

def get_directory_tree(path, indent=0, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = []

    # Get all items in the current directory
    try:
        items = os.listdir(path)
    except PermissionError:
        return  # Skip directories for which we don't have permission

    for item in items:
        item_path = os.path.join(path, item)

        # Skip directories that are in the exclude list
        if os.path.isdir(item_path) and item in exclude_dirs:
            continue

        # Print the current item with indentation
        print("    " * indent + "├── " + item)

        # If the item is a directory, recurse into it
        if os.path.isdir(item_path):
            get_directory_tree(item_path, indent + 1, exclude_dirs)

# Get the current directory
root_dir = os.getcwd()

# Define the directories to exclude
exclude_dirs = ['.git', 'node_modules', 'idea','datasets','venv']  # Add your directories here

# Print the root directory
print(root_dir)

# Generate the directory tree, excluding certain directories
get_directory_tree(root_dir, exclude_dirs=exclude_dirs)

# Wait for user input to prevent the window from closing
input("Press Enter to exit...")
