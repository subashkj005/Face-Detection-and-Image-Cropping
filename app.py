from flask import Flask, jsonify, render_template, request, send_file
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

app = Flask(__name__, template_folder='.')
app.static_folder = 'static'

def detect_and_crop_faces(image_data):
    # Convert image data to NumPy array
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert image to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        # Assume the first detected face is the main face
        x, y, w, h = faces[0]

        # Calculate the size for the square crop
        max_dim = max(w, h)
        new_size = int(2 * max_dim)

        # Calculate the coordinates for the square crop
        crop_x = max(0, x - (new_size - w) // 2)
        crop_y = max(0, y - (new_size - h) // 2)

        # Perform the square crop
        cropped_img = img[crop_y:crop_y+new_size, crop_x:crop_x+new_size]

        # Convert the cropped image to PIL Image format
        pil_image = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        
        # Save the PIL Image to BytesIO
        output_io = BytesIO()
        pil_image.save(output_io, format='JPEG')
        output_io.seek(0)

        return output_io
    else:
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' in request.files:
        image = request.files['image'].read()
        processed_image = detect_and_crop_faces(image)
        print('processed_image = ', processed_image)

        if processed_image is not None:
            image =  send_file(processed_image, mimetype='image/jpeg')
            return image
        else:
            return jsonify({"message": "Image processing failed."}), 200

    return jsonify({"message": "Image processing failed."})

if __name__ == '__main__':
    app.run(debug=True)
