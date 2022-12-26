import socket
import json
import os
import inspect
import sys
from distutils.util import execute

from flask import Flask, flash, request, redirect, url_for, render_template, send_file
from flask_cors import CORS, cross_origin

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from predict import predictStart
import models
import info

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Files')
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def base_route():
    return 'Server runs on port %s' % request.host


@app.route("/upload", methods=["POST"])
@cross_origin()
def upload():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    # file = request.files['file']
    app.logger.info(request.files)
    upload_files = request.files.getlist('file')
    app.logger.info(upload_files)
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if not upload_files:
        print('No selected file')
        return redirect(request.url)
    for file in upload_files:
        original_filename = file.filename
        filename = original_filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file_list = os.path.join(UPLOAD_FOLDER, 'files.json')
        files = _get_files()
        files[filename] = original_filename
        with open(file_list, 'w') as fh:
            json.dump(files, fh)
    print('Upload succeeded')
    return ""


@app.route("/files", methods=["GET"])
@cross_origin()
def get_files():
    file_list = os.path.join(UPLOAD_FOLDER, 'files.json')
    json_val = []
    if os.path.exists(file_list):
        with open(file_list) as fh:
            json_val = json.load(fh)
    return json_val


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict_():
    print(os.path.dirname(os.path.abspath(__file__)))
    content = request.json
    participant_id = content["participant_id"]
    model_name = content["model"]
    print(model_name)
    print(participant_id)
    diagnosis = predictStart(participant_id, model_name)
    print(diagnosis)
    return {"asd":diagnosis}


def _get_files():
    file_list = os.path.join(UPLOAD_FOLDER, 'files.json')
    if os.path.exists(file_list):
        with open(file_list) as fh:
            return json.load(fh)
    return {}


if __name__ == '__main__':
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('localhost', 0))
    port = sock.getsockname()[1]
    sock.close()
    app.run(port=port)
