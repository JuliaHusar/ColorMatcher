import matplotlib
matplotlib.use('MacOSX')  # Use the TkAgg backend
from PIL import Image, UnidentifiedImageError
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import colorsys


def rgb_to_hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0  # Normalize RGB values to 0-1 range
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return float(h), float(s), float(v)


# Folder paths / Image Paths
personal_image_path = 'personal_images/test_image.png'
image_path = 'album_covers/laufey.jpeg'
album_folder_path = 'album_covers'
image_folder_path = 'personal_images'

#Initialize dictionaries for storing two sets of images
personal_image_dictionary = {}
album_dictionary = {}
album_color_map = {}
album_color_map_hsv = {}
personal_image_hsv = []
smallest_difference = {'filename': None, 'difference': float('inf')}



#Load folders and every image inside folder into hashmaps for further use
def load_maps(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and not filename.startswith('.'):
            try:
                Image.open(file_path)  # Try to open the file as an image
                if folder_path == 'album_covers':
                    album_dictionary[filename] = file_path
                else:
                    personal_image_dictionary[filename] = file_path
            except (IOError, UnidentifiedImageError):
                print(f'Error loading image {filename}')
                continue


load_maps(album_folder_path)
load_maps(image_folder_path)

# personal_image = Image.open(personal_image_path)
img = Image.open(image_path)
cv2Image = cv2.imread(personal_image_path)
cv2PersonalImage = cv2.imread(personal_image_path)
img = img.resize((100, 100))  # Resize for faster processing
img = img.convert('RGB')  # Ensure it's in RGB format


def calculate_hsv_differences(hsv1, hsv2):
    hsv1 = np.array(hsv1, dtype=float)
    hsv2 = np.array(hsv2, dtype=float)
    difference = np.linalg.norm(hsv1 - hsv2)
    return difference


def analyze_personal_picture(personal_picture_path):
    # Preprocess image

    personal_picture = Image.open(personal_picture_path)
    personal_picture = personal_picture.resize((100, 100))
    personal_picture = personal_picture.convert('RGB')
    pixels = np.array(personal_picture)
    pixels = pixels.reshape(-1, 3)  # Flatten to a list of RGB values

    # Calculate k-means
    k_means = KMeans(n_clusters=3, random_state=0)
    k_means.fit(pixels)
    cluster_centers = k_means.cluster_centers_
    dominant_colors_rgb = cluster_centers.astype(int)

    # Create category for hsv values after conversion
    dominant_colors_hsv = []
    for dominant_color in dominant_colors_rgb:
        hsv_color = rgb_to_hsv(dominant_color[0], dominant_color[1], dominant_color[2])
        dominant_colors_hsv.append(hsv_color)

    return dominant_colors_hsv


personal_image_hsv.append(analyze_personal_picture(personal_image_path))
print("personal image hsv", personal_image_hsv)


def iterate_images(personal_image_HSV):
    for filename in album_dictionary:
        album_color_map[filename] = filename
        album_color_map_hsv[filename] = filename
    for filename in album_dictionary:
        extract_colors(filename, album_dictionary[filename])
    for album_hsv in album_color_map_hsv:
        difference = calculate_hsv_differences(personal_image_HSV, album_color_map_hsv[album_hsv])
        if difference < smallest_difference['difference']:
            smallest_difference['filename'] = album_hsv
            smallest_difference['difference'] = float(difference)
    print(smallest_difference)

# Extracts most prominent colors out of an image in rgb format, then pasting them in album color map
def extract_colors(filename, path_to_filename):
    processed_image = Image.open(path_to_filename)
    processed_image = processed_image.resize((100, 100))  # Resize for faster processing
    processed_image = processed_image.convert('RGB')  # Ensure it's in RGB format
    pixels = np.array(processed_image)
    pixels = pixels.reshape(-1, 3)  # Flatten to a list of RGB values
    # Apply KMeans clustering to find dominant colors
    kMeans = KMeans(n_clusters=3,
                    random_state=0)  # 5 clusters of dominant colors are created represented by 1 dimensional matrices
    kMeans.fit(pixels)
    cluster_centers = kMeans.cluster_centers_
    # Convert cluster centers to integers (RGB values)
    dominant_colors_rgb = cluster_centers.astype(int)
    dominant_colors_hsv = []
    for dominant_color in dominant_colors_rgb:
        hsv_color = rgb_to_hsv(dominant_color[0], dominant_color[1], dominant_color[2])
        dominant_colors_hsv.append(hsv_color)

    #print(dominant_colors_hsv)
    album_color_map[filename] = dominant_colors_rgb
    album_color_map_hsv[filename] = dominant_colors_hsv

    # Visualize the image pixels in 3D RGB space
    fig = plt.figure(figsize=(12, 8))
    figPlot = fig.add_subplot(111, projection='3d')

    # Normalize pixel values for proper coloring
    normalized_pixels = pixels / 255.0
    figPlot.scatter(pixels[:, 0], pixels[:, 1], pixels[:, 2], c=normalized_pixels, s=1)

    # Plot the cluster centers (dominant colors) on the scatter plot
    figPlot.scatter(
        cluster_centers[:, 0],
        cluster_centers[:, 1],
        cluster_centers[:, 2],
        c=cluster_centers / 255.0,
        s=200,
        marker='X',
        edgecolor='black',
        label='Dominant Colors'
    )

    figPlot.set_xlabel("Red")
    figPlot.set_ylabel("Green")
    figPlot.set_zlabel("Blue")
    figPlot.set_title("Scatter Plot of Image Pixels in RGB Space with Dominant Colors")
    figPlot.legend()

    # plt.show()


iterate_images(personal_image_hsv[0])

haar_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_fullbody.xml')
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

if body_cascade.empty():
    raise IOError('Failed to load Haar cascade file')

gray = cv2.cvtColor(cv2Image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
upper_body = upper_body_cascade.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=10, minSize=(100, 100))
body = body_cascade.detectMultiScale(gray, scaleFactor=1.001, minNeighbors=5, minSize=(100, 100))


def display_image(variable_name):
    for (x, y, w, h) in variable_name:
        cv2.rectangle(cv2Image, (x, y), (x + w, y + h), (255, 0, 0), 2)


display_image(body)

# Display the output
cv2.imshow('Body Detection', cv2Image)
cv2.waitKey(0)
cv2.destroyAllWindows()
