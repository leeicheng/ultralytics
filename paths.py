import os

def assemble_input_and_output_paths(root_dir, src_dir_keyword, dest_dir_keyword, src_extension, dest_extension):
    """
    Search for all files with the specified source extension in the root directory,
    then generate corresponding output paths by replacing parts of the directory name.

    Args:
        root_dir (str): The root directory to search for files.
        src_dir_keyword (str): The keyword in the source directory name to be replaced.
        dest_dir_keyword (str): The keyword to replace with in the destination directory name.
        src_extension (str): The extension of the source files (e.g., ".csv" or ".mp4").
        dest_extension (str): The extension for the output files (e.g., ".json").

    Returns:
        tuple: A tuple containing two lists:
            - A list of input file paths.
            - A list of corresponding output file paths.
    """

    file_paths = []
    output_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(src_extension):
                # 取得讀取路徑
                file_path = os.path.join(dirpath, filename)
                file_paths.append(file_path)
                # 產生輸出路徑
                output_dir = dirpath.replace(src_dir_keyword, dest_dir_keyword)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                # 取得檔名的主體與副檔名
                base, ext = os.path.splitext(filename)
                ext = ext.replace(src_extension, dest_extension)
                if dest_dir_keyword.lower() not in base.lower():
                    # 組合新的檔名，例如：XXX -> XXX_dir
                    filename = f"{base}_{dest_dir_keyword}{ext}"
                else :
                    filename = f"{base}{ext}"
                out_path = os.path.join(output_dir, filename)
                output_paths.append(out_path)
    return file_paths,output_paths

def find_all_csv_files(root_dir):
    """
    Find all CSV files in the given root directory and generate output paths in JSON format.
    """
    return assemble_input_and_output_paths(root_dir, "csv", "ball", ".csv", ".json")

def find_all_video_files(root_dir):
    """
    Find all video files (with .mp4 extension) in the given root directory and generate output paths in JSON format.
    """
    return assemble_input_and_output_paths(root_dir, "video", "pose", ".mp4", ".json")

def create_crop_region_paths(root_dir):
    """
    Create output paths for video files using an alternative destination directory keyword.
    """
    return assemble_input_and_output_paths(root_dir, "video", "area", ".mp4", ".json")

def create_multi_output_paths(root_dir):
    """
    Create output paths for video files using an alternative destination directory keyword.
    """
    return assemble_input_and_output_paths(root_dir, "video", "multi", ".mp4", ".json")

def create_preview_video_paths(root_dir):
    return assemble_input_and_output_paths(root_dir, "video", "preview", ".mp4", ".mp4")