import requests
import json
from PIL import Image, ImageDraw
import os

def detect_face(image_path, api_key):
    url = "https://face-analyzer1.p.rapidapi.com/estimate/age-gender"

    headers = {
        "x-rapidapi-host": "face-analyzer1.p.rapidapi.com",
        "x-rapidapi-key": api_key
    }

    with open(image_path, "rb") as image_file:
        files = {"image": image_file}
        response = requests.post(url, headers=headers, files=files)

    if response.status_code == 200:
        return json.loads(response.text)
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def draw_rectangles(image_path, boxes):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for box in boxes:
        x, y, width, height = box
        draw.rectangle([x, y, width, height], outline="red", width=2)

    return image

def main():
    image_path = "/home/tamnv/Downloads/jixiao-huang-I1Plc-FAAnQ-unsplash.jpg"
    api_key = "e5894aae1bmsh56f932fdebc726dp11fec1jsn6f2ebad3f61c"

    for i in range(100):
        print(i)
        result = detect_face(image_path, api_key)

        if result and result["success"]:
            boxes = [face["box"] for face in result["data"]]

            print(f"Image with detected faces saved as")
        else:
            print("Face detection failed or no faces detected.")

if __name__ == "__main__":
    main()