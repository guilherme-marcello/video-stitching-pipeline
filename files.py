import re

def extract_number(filename):
    """
    Extracts the number from a filename following the pattern: 
    'img_<number>.jpg' or 'yolo_<number>.mat'

    Args:
    filename: The name of the file.

    Returns:
    The extracted number as a string, or empty string if no number is found.
    """
    match = re.search(r'_(\d+)\.', filename)
    if match:
        return str(match.group(1))
    else:
        return str("")