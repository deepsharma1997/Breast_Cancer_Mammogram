import os
import pandas as pd
import numpy as np
import nibabel as nib
import pydicom 
import matplotlib.pyplot as plt
import skimage
import cv2
from skimage import io


def crop_from_up_down(image, top_percent, bottom_percent):
    img_array = image.pixel_array
    # Get image dimensions
    height, width = img_array.shape[:2]
    
    # Calculate cropping boundaries
    top_crop = int(height * top_percent / 100)
    bottom_crop = int(height * bottom_percent / 100)
    
    # Crop the image
    out_array = img_array[top_crop:height - bottom_crop, :]
    image.PixelData = out_array.tobytes()
    image.Rows, image.Columns = out_array.shape
    return image


def mask_breast_region_only(image):
    image_arr = image.pixel_array
    if image.PhotometricInterpretation == "MONOCHROME1":
        image_arr = np.max(image_arr) - image_arr
    else:
        image_arr = image_arr - np.min(image_arr)
    
    if np.max(image_arr) != 0:
        image_arr = image_arr / np.max(image_arr)
        image_arr = (image_arr * 255).astype(np.uint8)

    # Binarize the image
    bin_pixels = cv2.threshold(image_arr, 0, 255, cv2.THRESH_BINARY)[1]
    

    # Make contours around the binarized image, keep only the largest contour
    contours, _ = cv2.findContours(bin_pixels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    
    contour = max(contours, key=cv2.contourArea)

    # Create a mask from the largest contour
    mask = np.zeros(image_arr.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    mask = mask/255
    #plt.imshow(mask)
    #plt.show()
    
    # Use bitwise_and to get masked part of the original image
    #out = cv2.bitwise_and(image.pixel_array,mask)
    out_array = np.where(mask, image.pixel_array, 0)
    image.PixelData = out_array.tobytes()   
    return image


def create_square_mask_with_padding(mask, padding):
    """
    Create a square-shaped mask around the labeled region with padding.
    Parameters:
    - mask (numpy.ndarray): The binary mask with labeled regions.
    - padding (int): The padding to add around the square mask.
    Returns:
    - numpy.ndarray: A new binary mask with a square bounding box around the labeled region including padding.
    """
    # Ensure the mask is binary
    # mask = mask.astype(bool)
    # Find the bounding box of the labeled region
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        raise ValueError("No labeled region found in the mask")
        
    r_min, r_max = np.argmax(rows), len(rows) - np.argmax(np.flip(rows))
    c_min, c_max = np.argmax(cols), len(cols) - np.argmax(np.flip(cols))

    # Determine the size of the square bounding box
    height = r_max - r_min
    width = c_max - c_min
    max_dim = max(height, width)

    # Calculate the center of the original bounding box
    center_y = (r_min + r_max) // 2
    center_x = (c_min + c_max) // 2

    # Determine the new bounding box to be a square with padding
    new_r_min = max(center_y - max_dim // 2 - padding, 0)
    new_r_max = min(center_y + max_dim // 2 + padding + max_dim % 2, mask.shape[0])
    new_c_min = max(center_x - max_dim // 2 - padding, 0)
    new_c_max = min(center_x + max_dim // 2 + padding + max_dim % 2, mask.shape[1])

    # Create the square mask with padding
    square_mask = np.zeros_like(mask)
    square_mask[new_r_min:new_r_max, new_c_min:new_c_max] = True

    return square_mask

def apply_mask_on_image(image, mask):
    # Ensure the mask is binary
    mask[mask==255] = 1
    
    # Apply the mask to the image
    masked_image = image*mask
    return masked_image

def apply_mask_on_mask(image, mask):
    # Ensure the mask is binary
    image[image==0] = 120

    
    # Apply the mask to the image
    masked_image = image*mask
    return masked_image

def crop_non_zero_region_with_padding(image, padding):
    """
    Crop the non-zero region from the image with additional padding.

    Parameters:
    - image (numpy.ndarray): The input image.
    - padding (int): The padding to add around the non-zero region.

    Returns:
    - numpy.ndarray: The cropped image with padding.
    """
    # Ensure the image is in grayscale (if it's not already)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find the bounding box of the non-zero region
    rows = np.any(image > 0, axis=1)
    cols = np.any(image > 0, axis=0)

    if not np.any(rows) or not np.any(cols):
        raise ValueError("No non-zero region found in the image")

    r_min, r_max = np.argmax(rows), len(rows) - np.argmax(np.flip(rows))
    c_min, c_max = np.argmax(cols), len(cols) - np.argmax(np.flip(cols))

    # Add padding
    r_min = max(r_min - padding, 0)
    r_max = min(r_max + padding, image.shape[0])
    c_min = max(c_min - padding, 0)
    c_max = min(c_max + padding, image.shape[1])

    # Crop the image
    cropped_image = image[r_min:r_max, c_min:c_max]
    return cropped_image

"MONOCHROME1 indicates that the greyscale ranges from bright to dark with ascending pixel values, whereas MONOCHROME2 ranges from dark to bright with ascending pixel values"

def min_max_scale(image_array):
    # Convert the image array to float32 for precision in calculations
    image_array = image_array.astype(np.float32)
    
    # Find the min and max values in the image array
    min_val = np.min(image_array)
    max_val = np.max(image_array)
    
    # Apply min-max normalization
    normalized_image_array = (image_array - min_val) / (max_val - min_val)
    return normalized_image_array

def right_orient_mammogram(image):
    left_nonzero = cv2.countNonZero(image[:, 0:int(image.shape[1]/2)])
    right_nonzero = cv2.countNonZero(image[:, int(image.shape[1]/2):])
    
    if(left_nonzero < right_nonzero):
        image = cv2.flip(image, 1)

    return image




calc_train = pd.read_csv("D:/plaksha/code/mammo/all_info_final_excel/calc_train.csv")
calc_test = pd.read_csv("D:/plaksha/code/mammo/all_info_final_excel/calc_test.csv")
mass_train = pd.read_csv("D:/plaksha/code/mammo/all_info_final_excel/mass_train.csv")
mass_test = pd.read_csv("D:/plaksha/code/mammo/all_info_final_excel/mass_test.csv")


output_image_dir = 'D:/Plaksha/code/mammo/uint8/all_info_final_cm_23_5/full_image'
output_mask_dir = 'D:/Plaksha/code/mammo/uint8/all_info_final_cm_23_5/full_mask'
output_crop_image_dir = 'D:/Plaksha/code/mammo/uint8/all_info_final_cm_23_5/crop_image'
output_crop_mask_dir = 'D:/Plaksha/code/mammo/uint8/all_info_final_cm_23_5/crop_mask'
output_breast_image_dir = 'D:/Plaksha/code/mammo/uint8/all_info_final_cm_23_5/breast_image'
output_clahe_image_dir = 'D:/Plaksha/code/mammo/uint8/all_info_final_cm_23_5/clahe_image'

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)
os.makedirs(output_crop_image_dir, exist_ok=True)
os.makedirs(output_crop_mask_dir, exist_ok=True)
os.makedirs(output_breast_image_dir, exist_ok=True)
os.makedirs(output_clahe_image_dir, exist_ok=True)

variables = {'calc_train':calc_train, 'calc_test':calc_test, 'mass_train':mass_train, 'mass_test':mass_test}
for name, value in variables.items():
    print(name)
    
    
    for index, row in value.iterrows():
        if index >-1 and row['img_label']== 'BENIGN' and row['assessment']==2:


            print(f"Index: {index}")
            #print(f"img: {row['img']}, mask: {row['mask']}, img_path: {row['img_path']}, mask_path: {row['mask_path']}, crop_path: {row['crop_path']} , img_label: {row['img_label']}")
            image = pydicom.dcmread(row['img_path'])
            mask = pydicom.dcmread(row['mask_path'])
            crop = pydicom.dcmread(row['crop_path'])
            

            sample_crop = pydicom.dcmread(row['crop_path'])
            sample_mask = pydicom.dcmread(row['mask_path'])
            


            image = crop_from_up_down(image, 4,4)
            '''plt.imshow(image.pixel_array)
            plt.show()
            image = mask_breast_region_only(image)
            plt.imshow(image.pixel_array)
            plt.show()'''

            mask = crop_from_up_down(mask, 4,4)
            """pixels = cv2.countNonZero(mask.pixel_array)
            image_area = mask.pixel_array.shape[0] * mask.pixel_array.shape[1]
            area_ratio = (pixels / image_area) * 100
            print(area_ratio)"""
            

            '''fig, (ax1,ax2) = plt.subplots(1,2,figsize = (12,12))
            fig.suptitle(row['img_label'], fontsize=12, fontweight='bold')
            ax1.imshow(image.pixel_array, cmap = 'gray')
            ax1.set_title("Normal, max= " + str(np.max(image.pixel_array)) + ", min =" + str(np.min(image.pixel_array)), fontsize=20)
            ax2.imshow(mask.pixel_array, cmap = 'gray')
            ax2.set_title("HE, max= " + str(np.max(mask.pixel_array)) + ", min =" + str(np.min(mask.pixel_array)), fontsize=20)
            plt.show()'''

            
            image_arr = image.pixel_array
            mask_arr = mask.pixel_array
            crop_arr = crop.pixel_array


            breast = mask_breast_region_only(image)
            breast_arr = breast.pixel_array

            image_arr = cv2.normalize(image_arr, None, 0, 255, cv2.NORM_MINMAX)
            image_arr = np.uint8(image_arr)

            breast_arr = cv2.normalize(breast_arr, None, 0, 255, cv2.NORM_MINMAX)
            breast_arr = np.uint8(breast_arr)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
            clahe_arr = clahe.apply(breast_arr)

            #normalized_image = min_max_scale(image_arr)*255
            '''normalized_image = cv2.normalize(image_arr, None, 0, 255, cv2.NORM_MINMAX)
            normalized_image = np.uint8(normalized_image)

            
            sobel_x = cv2.Sobel(normalized_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(normalized_image, cv2.CV_64F, 0, 1, ksize=3)
            sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
            sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))
        
            normalized_image = cv2.addWeighted(normalized_image, 0.85, sobel_combined, 0.15, 0)

            HE = cv2.equalizeHist(normalized_image) 
        
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            clahe_image = clahe.apply(HE)
            '''
            mask_box = create_square_mask_with_padding(mask_arr, 30)  
            

            masked_image = apply_mask_on_image(clahe_arr, mask_box)
            masked_mask = apply_mask_on_mask(mask_arr, mask_box)
            crop_image = crop_non_zero_region_with_padding(masked_image, 0)
            crop_mask = crop_non_zero_region_with_padding(masked_mask, 0)

            #print(np.unique(crop_mask))
            crop_mask[crop_mask==120] = 0

            #print(np.unique(mask_arr))
            mask_arr[mask_arr==120] = 0


            '''fig, axes = plt.subplots(1, 5, figsize=(10, 5))

            # Display the first image
            axes[0].imshow(image_arr)
            axes[0].axis('off')  # Turn off axis

            # Display the second image
            axes[1].imshow(breast_arr)
            axes[1].axis('off')  # Turn off axis

            axes[2].imshow(mask_arr)
            axes[2].axis('off')  # Turn off axis

            # Display the second image
            axes[3].imshow(crop_image)
            axes[3].axis('off')  # Turn off axis

            # Display the second image
            axes[4].imshow(crop_mask)
            axes[4].axis('off')  # Turn off axis

            # Show the plot
            plt.show()'''

        
            out_path = os.path.join(output_image_dir,name, row['img_label'])
            os.makedirs(out_path, exist_ok=True)
            file_name = row['img'] + '.png'
            output_path = os.path.join(out_path , file_name)
            cv2.imwrite(output_path, image_arr)

            out_path = os.path.join(output_breast_image_dir,name, row['img_label'])
            os.makedirs(out_path, exist_ok=True)
            file_name = row['img'] + '.png'
            output_path = os.path.join(out_path , file_name)
            cv2.imwrite(output_path, breast_arr)

            out_path = os.path.join(output_clahe_image_dir,name, row['img_label'])
            os.makedirs(out_path, exist_ok=True)
            file_name = row['img'] + '.png'
            output_path = os.path.join(out_path , file_name)
            cv2.imwrite(output_path, clahe_arr)

            out_path = os.path.join(output_crop_image_dir,name, row['img_label'])
            os.makedirs(out_path, exist_ok=True)
            file_name = row['mask'] + '.png'
            output_path = os.path.join(out_path , file_name)
            cv2.imwrite(output_path, crop_image)

            out_path = os.path.join(output_mask_dir,name, row['img_label'])
            os.makedirs(out_path, exist_ok=True)
            file_name = row['mask'] + '.png'
            output_path = os.path.join(out_path , file_name)
            cv2.imwrite(output_path, mask_arr)

            out_path = os.path.join(output_crop_mask_dir,name, row['img_label'])
            os.makedirs(out_path, exist_ok=True)
            file_name = row['mask'] + '.png'
            output_path = os.path.join(out_path , file_name)
            cv2.imwrite(output_path, crop_mask)


            
