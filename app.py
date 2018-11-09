"""
Flask Serving

This file is a sample flask app that can be used to test your model with an API.

This app does the following:
    - Handles uploads and looks for an image file send as "file" parameter
    - Stores the image at ./images dir
    - Invokes ffwd_to_img function from evaluate.py with this image
    - Returns the output file generated at /output

Additional configuration:
    - You can also choose the checkpoint file name to use as a request parameter
    - Parameter name: checkpoint
    - It is loaded from /input
"""
import os
import base64
from io import BytesIO

from flask import Flask, send_file, request, make_response, render_template

from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename

import predict

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('serving_template.html')

	
@app.route('/', methods=["POST"])
def predict_plants():
    """
    Take the input image and style transfer it
    """
    # check if the post request has the file part
    input_file = request.files.get('file')
    if not input_file:
        return BadRequest("File not present in request")

    filename = secure_filename(input_file.filename)
    if filename == '':
        return BadRequest("File name is not present in request")
    if not allowed_file(filename):
        return BadRequest("Invalid file type")

        # # Save Image to process
    input_buffer = BytesIO()
    output_buffer = BytesIO()
    input_file.save(input_buffer)

    img = predict.get_plants(input_buffer)
    img.save(output_buffer, format="JPEG")
    img_str = base64.b64encode(output_buffer.getvalue())

    response = make_response(img_str)
    response.headers.set('Content-Type', 'image/jpeg')
    return response

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run(host='0.0.0.0')
