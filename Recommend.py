import cv2
import numpy as np
from sklearn.cluster import KMeans
import tensorflow as tf

# Colors dictionary
colors = {
    "Light Blue": (((150, 200), (200, 255), (220, 255)), "White, Brown"),
    "Blue": (((0, 50), (0, 50), (150, 255)), "White, Brown"),
    "Dark Blue": (((0, 30), (0, 30), (80, 150)), "Brown, Black"),
    "Light Red": (((220, 255), (150, 200), (150, 220)), "White, Black"),
    "Red": (((200, 255), (0, 50), (0, 50)), "White, Black"),
    "Dark Red": (((80, 150), (0, 30), (0, 30)), "Black"),
    "Light Green": (((100, 200), (200, 255), (100, 200)), "Brown, Black"),
    "Green": (((0, 50), (100, 200), (0, 50)), "Brown, Black"),
    "Dark Green": (((0, 30), (60, 120), (0, 30)), "Black"),
    "Light Yellow": (((220, 255), (220, 255), (100, 200)), "White, Brown"),
    "Yellow": (((200, 255), (200, 255), (0, 50)), "White, Brown"),
    "Dark Yellow": (((150, 220), (150, 220), (0, 30)), "Brown"),
    "Light Purple": (((200, 255), (150, 220), (200, 255)), "Brown, Black"),
    "Purple": (((100, 200), (0, 100), (100, 200)), "Brown, Black"),
    "Dark Purple": (((50, 80), (0, 50), (50, 100)), "Black , White"),
    "Light Orange": (((220, 255), (180, 220), (80, 150)), "Brown, Black"),
    "Orange": (((200, 255), (100, 200), (0, 50)), "Brown, Black"),
    "Dark Orange": (((150, 220), (50, 100), (0, 30)), "Black"),
    "Light Pink": (((220, 255), (150, 200), (180, 220)), "White, Brown"),
    "Pink": (((200, 255), (50, 150), (100, 200)), "White, Brown"),
    "Dark Pink": (((150, 220), (0, 50), (80, 150)), "Black"),
    "Light Brown": (((200, 255), (150, 200), (100, 150)), "Brown, Black"),
    "Brown": (((100, 160), (40, 80), (0, 60)), "Black"),
    "Dark Brown": (((40, 90), (20, 60), (0, 40)), "Black"),
    "Light Gray": (((200, 255), (200, 255), (200, 255)), "White, Brown"),
    "Gray": (((100, 200), (100, 200), (100, 200)), "White, Brown"),
    "Dark Gray": (((50, 100), (50, 100), (50, 100)), "Brown, Black"),
    "Light Cyan": (((200, 255), (220, 255), (220, 255)), "White, Brown"),
    "Cyan": (((0, 50), (200, 255), (200, 255)), "White, Brown"),
    "Dark Cyan": (((0, 30), (100, 150), (100, 150)), "Black"),
    "Light Magenta": (((220, 255), (150, 200), (220, 255)), "White, Brown"),
    "Magenta": (((200, 255), (0, 50), (200, 255)), "White, Brown"),
    "Dark Magenta": (((100, 150), (0, 30), (100, 150)), "Black"),
    "Light Teal": (((150, 200), (200, 255), (200, 255)), "White, Brown"),
    "Teal": (((0, 50), (150, 200), (150, 200)), "Brown, Black"),
    "Dark Teal": (((0, 30), (80, 120), (80, 120)), "Black"),
    "Light Beige": (((220, 255), (200, 255), (180, 220)), "White, Brown"),
    "Beige": (((180, 220), (150, 200), (100, 150)), "Brown, Black"),
    "Dark Beige": (((120, 180), (100, 150), (50, 100)), "Black"),
    "Maroon": (((100, 150), (0, 50), (0, 50)), "Black"),
    "Taupe": (((100, 150), (70, 120), (50, 100)), "Brown, Black"),
    "Olive": (((50, 100), (50, 100), (0, 50)), "Black"),
}

def load_pretrained_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

#Image Preprocessing
def preprocess_image(image_path, target_size=(128, 128)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Couldn’t load image at {image_path}. Check the path and file!")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, target_size)
    image_normalized = image_resized / 255.0
    image_input = np.expand_dims(image_normalized, axis=0)
    return image, image_resized

# predicting segmented mask
def segment_image(model, image_input):
    segmentation_mask = model.predict(image_input, verbose=0)
    segmentation_mask = np.argmax(segmentation_mask[0], axis=-1) if segmentation_mask.shape[-1] > 1 else segmentation_mask[0] > 0.5
    return segmentation_mask.astype(np.uint8)



"""steps:

Resizes mask to match original image size.

Selects pixels where mask = 1 (shirt area).

Uses KMeans with 2 clusters to find main color.

Returns RGB of the largest cluster (dominant color)."""
def get_dominant_color(original_image, segmentation_mask):
    mask_resized = cv2.resize(segmentation_mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    segmented_pixels = original_image[mask_resized > 0]
    if len(segmented_pixels) == 0:
        return None
    # Use 2 clusters to separate major colors (e.g., yellow and white)
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(segmented_pixels)
    # Get the cluster with the most pixels
    labels = kmeans.predict(segmented_pixels)
    dominant_cluster = np.argmax(np.bincount(labels))
    dominant_color = kmeans.cluster_centers_[dominant_cluster]
    return dominant_color.astype(int)


#Match to Color Name + Skin Tone
def match_color_name(rgb_values):
    if rgb_values is None:
        return None
    r, g, b = rgb_values
    for color_name, (rgb_range, skin_tones) in colors.items():
        (r_min, r_max), (g_min, g_max), (b_min, b_max) = rgb_range
        if r_min <= r <= r_max and g_min <= g <= g_max and b_min <= b <= b_max:
            return color_name, skin_tones
    min_distance = float('inf')
    matched_color = None
    for color_name, (rgb_range, skin_tones) in colors.items():
        (r_min, r_max), (g_min, g_max), (b_min, b_max) = rgb_range
        r_mid, g_mid, b_mid = (r_min + r_max) / 2, (g_min + g_max) / 2, (b_min + b_max) / 2
        distance = np.sqrt((r - r_mid)**2 + (g - g_mid)**2 + (b - b_mid)**2)
        if distance < min_distance:
            min_distance = distance
            matched_color = (color_name, skin_tones)
    return matched_color

def process_image_and_get_color(image_path, model_path):
    model = load_pretrained_model(model_path)
    original_image, image_resized = preprocess_image(image_path)
    image_input = np.expand_dims(image_resized / 255.0, axis=0)
    segmentation_mask = segment_image(model, image_input)

    dominant_color = get_dominant_color(image_resized, segmentation_mask)

    mask_resized = cv2.resize(segmentation_mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    segmented_region = np.zeros_like(original_image)
    if dominant_color is not None:
        segmented_region[mask_resized > 0] = [dominant_color[0], dominant_color[1], dominant_color[2]]

    original_display = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    mask_display = cv2.cvtColor((mask_resized * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    segmented_display = cv2.cvtColor(segmented_region.astype(np.uint8), cv2.COLOR_RGB2BGR)

    combined = np.hstack((original_display, mask_display, segmented_display))
    cv2.imwrite("output_combined.jpg", combined)
    print("Combined output saved as 'output_combined.jpg'")

    if dominant_color is not None:
        rgb_dict = {"R": int(dominant_color[0]), "G": int(dominant_color[1]), "B": int(dominant_color[2])}
        color_name, skin_tones = match_color_name(dominant_color)
        return {
            "RGB": rgb_dict,
            "Color Name": color_name,
            "Matched Skin Tones": skin_tones,
            "Visual Output": "Check 'output_combined.jpg' for Original Image, Predicted Mask, and Segmented Region"
        }
    else:
        return "No segmented region found."

if __name__ == "__main__":
    image_path = "/content/Screenshot 2025-04-08 074146.png"  # Update with your uploaded image path
    model_path = "/content/alp3.keras"

    try:
        result = process_image_and_get_color(image_path, model_path)
        print("Result:", result)
    except ValueError as e:
        print(f"Error: {e}")


from google.colab.patches import cv2_imshow
cv2_imshow(cv2.imread("output_combined.jpg"))
