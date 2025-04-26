# Shirt--Segmentation-Out-Fit-Recommendation
This Project is about Shirt Segmentation and extracting shirt Dominant color &amp; Performing Outfit Recommendation From Extracted Color 

Project Overview:
Shirt Segmentation, Dominant Color Extraction, and Outfit Recommendation

This project focuses on building a pipeline that segments shirts from images, extracts their dominant colors, and uses this information to recommend matching outfits.

1. Shirt Segmentation :

The first step involves detecting and isolating the shirt region from an input image. Using advanced deep learning models  CNN, U-Net, the system accurately separates the shirt from the rest of the image (background, pants, accessories, etc.). This ensures that only the relevant portion — the shirt — is analyzed for color features.

2. Dominant Color Extraction :
   
Once the shirt area is segmented, color analysis is performed. Using clustering algorithm K-Means the system identifies the most prominent colors present in the shirt. This step involves:

      Converting the image to a suitable color space RGB .

      Applying clustering to group similar colors.

      Selecting the cluster centroid with the highest frequency as the "dominant color."

      The extracted dominant color is represented in a standard format (e.g., HEX, RGB) for further use.
      
      
3. Rule-Based Color Matching :
   
        A small, predefined dataset maps shirt color categories to:

       Specific RGB value ranges

       Recommended complementary skin color.

Example structure:

colors = {

    "Light Blue": (((150, 200), (200, 255), (220, 255)), "White, Brown"),
    
    "Blue": (((0, 50), (0, 50), (150, 255)), "White, Brown"),
    
    "Dark Blue": (((0, 30), (0, 30), (80, 150)), "Brown, Black"),
    
    ...
}

Matching Process:

Compare the extracted dominant RGB value against all predefined ranges.

Find the closest matching shirt color category.

Recommend the corresponding outfit to suitable skin color options.

Output:
Combined output saved as 'output_combined.jpg'
Result: {'RGB': {'R': 75, 'G': 50, 'B': 70}, 'Color Name': 'Dark Purple', 'Matched Skin Tones': 'Black , White', 'Visual Output': "Check 'output_combined.jpg' for Original Image, Predicted Mask, and Segmented Region"}

![WhatsApp Image 2025-04-07 at 13 17 14_f52eb1ea](https://github.com/user-attachments/assets/c0c90a2d-10a2-496c-bf59-22e2b3fca142)

