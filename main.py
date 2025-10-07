from split_image import split_image
from detection import ship_detection
from model.model_to_int8 import int8_model
import shutil

import os
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_IMG_DIR = os.path.join(BASE_DIR, "input_image")
OUTPUT_IMG_DIR = os.path.join(BASE_DIR, "output_image")
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
TRANSFER_IMG_DIR = os.path.join(BASE_DIR, "transfer_image")

def scan_input(input_dir = INPUT_IMG_DIR):
    """
    Scan a input images path and return their paths.

    Args:
        input_dir : str
            Path containing images. Defaults to INPUT_IMG_DIR.

    Returns:
        list of str
            A list of full file paths to all images found in the directory.
        """
    large_image_paths = []
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        large_image_paths.append(img_path)
    return large_image_paths

def detect_a_large_image(model, img_path, patch_size = 768, output_dir = OUTPUT_IMG_DIR):
    """
    Detect and write information of a large image to output_dir

    Args:
        model: YOLO model
        img_path: a large image path

    Returns:
        None
    Result:
        write information of a large image to output_dir with image path name as folder
        
        """
        
    a_large_image = cv2.imread(img_path)
    image_patches = split_image(a_large_image, patch_size)
    a_large_image_name_extend = os.path.basename(img_path) #image name with .jpg
    a_large_image_name = os.path.splitext(a_large_image_name_extend)[0] #img name
    
    output_of_a_large_image_folder = os.path.join(output_dir, a_large_image_name) #output_image/1/
    os.makedirs(output_of_a_large_image_folder, exist_ok=True)
    a_large_detect_infors = []
    for patch, offset in image_patches:
        patch_detect_infors = ship_detection(model, patch)
        for xyxy, confs in patch_detect_infors:
            for box, c in zip(xyxy, confs):
                x_min, y_min, x_max, y_max = box
                x_min, y_min, x_max, y_max = (
                    int(x_min + offset[0]),
                    int(y_min + offset[1]),
                    int(x_max + offset[0]),
                    int(y_max + offset[1]),
                )
                ship_infor = f"{x_min}, {y_min}, {x_max}, {y_max}, {c:.4f}"
                a_large_detect_infors.append(ship_infor)
    
    a_large_detect_infors_file = os.path.join(output_of_a_large_image_folder,"detection.txt") #output_image/1/detection.txt
    with open(a_large_detect_infors_file, "w") as f:
        for line in a_large_detect_infors:
            f.write(line + "\n")
    shutil.move(img_path, output_of_a_large_image_folder)
    
def scan_output(output_dir):
    index = 1
    for img_processed_name in scan_input(output_dir):
        img_processed_folder = os.path.join(output_dir, img_processed_name)
        with open(os.path.join(img_processed_folder, "detection.txt"), "r") as f:
            lines = f.readlines()
        ship_number = len(lines)
        print(f"{index} | {img_processed_name} | Number of ship: {ship_number}")
        index += 1

def retrieve_request(output_dir, retrieve_array, transfer_dir, all = False):
    if all == False:
        retrieve_index = 0
        index = 1
        for img_processed_name in scan_input(output_dir):
            if retrieve_array[retrieve_index] == index:
                img_processed_folder = os.path.join(output_dir, img_processed_name)
                shutil.move(img_processed_folder, transfer_dir)
                retrieve_index += 1
                if retrieve_index == len(retrieve_array): break
            index += 1
    else:
        for img_processed_name in scan_input(output_dir):
            img_processed_folder = os.path.join(output_dir, img_processed_name)
            shutil.move(img_processed_folder, transfer_dir)
        
def run():
    
    """
    Processing new input when call this function        
    """
    large_image_paths = scan_input(INPUT_IMG_DIR)
    if len(large_image_paths) == 0:
        print("Nothing to process!!!")
    else:
        model = int8_model()
        for large_image_path in large_image_paths:
            print("Processing: ", large_image_path)
            detect_a_large_image(model, large_image_path, 768, OUTPUT_IMG_DIR)
            
if __name__ == "__main__":
    run()
    while True:
        print("="*40)
        print("                MAIN MENU")
        print("="*40)
        print("1. Number of processed images")
        print("2. Check ship detection and retrieve some images")
        print("3. Retrieve all processed images")
        print("0. Exit!")
        
        print("\nYour choice:")
        print("="*40)
        try:
            op = int(input("Enter option: "))
        except ValueError:
            print("Invalid input! Please enter a number.")
            continue
        if op == 0:
            print("Exiting DONE")
            break
        elif op == 1:
            print("Number of processed image: ", len(os.listdir(OUTPUT_IMG_DIR)))
        elif op == 2:
            print("Listing images")
            scan_output(OUTPUT_IMG_DIR)
            user_input = str(input("Do you want retrieve some images, e.g: Enter 1 5 6 8 to collect images 1, 5, 6, and 8 or 0 to back Menu :"))
            if user_input == "0":
                continue
            else:
                numbers = list(map(int, user_input.split()))
                numbers.sort()
                retrieve_request(OUTPUT_IMG_DIR, numbers, TRANSFER_IMG_DIR)
        elif op == 3:
            retrieve_request(OUTPUT_IMG_DIR, None, TRANSFER_IMG_DIR, True)
            
        else:
            print("⚠️  Invalid option, please try again.")