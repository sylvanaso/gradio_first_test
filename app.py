import requests
import json
from PIL import Image
import numpy as np
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((256,256))  # Resize the image for consistency
    return img

def image_to_vector(img):
    img_array = np.array(img)
    img_vector = img_array.flatten().astype(np.float32) / 255.0  # Normalize pixel values
    return img_vector

def cosine_distance(vector1, vector2):
    return 1 - cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))[0][0]

def cos_similarity(image_url1,image_url2):
    # Replace these URLs with the URLs of your images
    #image_url1 = "http://5.imimg.com/data5/SELLER/Default/2023/6/314067221/HY/QZ/TW/28643928/picsart-23-02-11-08-44-47-891.jpg"
    #image_url2 = "http://5.imimg.com/data5/WZ/DB/AE/ANDROID-28643928/product-jpeg.jpg"

    # Load images from URLs
    image1 = load_image_from_url(image_url1)
    image2 = load_image_from_url(image_url2)

    # Convert images to vectors
    vector1 = image_to_vector(image1)
    vector2 = image_to_vector(image2)

    # Calculate cosine distance
    distance = cosine_distance(vector1, vector2)


    # Print the cosine distance
    return str(abs(round(distance,4))),f"""
        <div style='display: flex;'>
            <div style='flex: 1; text-align: center;'>
                <p>Image 1</p>
                <img src='{image_url1}' style='max-width: 90%; max-height: 90%; margin: auto; display: block;'>
            </div>
            <div style='flex: 1; text-align: center;'>
                <p>Image 2</p>
                <img src='{image_url2}' style='max-width: 90%; max-height: 90%; margin: auto; display: block;'>
            </div>
        </div>
    """

def cos():
    iface = gr.Interface(fn=cos_similarity,inputs=[gr.Textbox(label="URL 1"),gr.Textbox(label="URL 2")],outputs=[gr.Textbox(label="Cosine Distance"),"html"],title="Cosine Similarity API",allow_flagging="never",)
    iface.launch(server_name="0.0.0.0",server_port=8110)


if __name__ == "__main__":
    cos()

