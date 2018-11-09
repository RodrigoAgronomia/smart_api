#export FLASK_APP=app.py
#flask run --host=0.0.0.0

import os
from flask import Flask, send_file, request, make_response, render_template

from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename

import predict

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
	return render_template('serving_template.html')

	
@app.route('/image', methods=["POST"])
def predict_plants_web():
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

	input_filepath = os.path.join('uploads', 'test', filename)
	input_file.save(input_filepath)
	output_filepath = os.path.join('results', 'test', 'msk_' + filename)
	img_str = predict.get_plants(input_filepath, output_filepath)

	response = make_response(img_str)
	response.headers.set('Content-Type', 'image/jpeg')
	return response
	
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

	input_filepath = os.path.join('uploads', 'test', filename)
	input_file.save(input_filepath)
	output_filepath = os.path.join('results', 'test', 'msk_' + filename)
	img_str = predict.get_plants(input_filepath, output_filepath)

	return send_file(output_filepath, mimetype='image/jpg')

def allowed_file(filename):
	return '.' in filename and \
		filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
	app.run(host='0.0.0.0')
