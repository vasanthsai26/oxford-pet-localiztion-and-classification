import os
import utils.config as config

from utils.inference import make_predictions
from flask import Flask,render_template,request,send_from_directory,session
from flask_dropzone import Dropzone
from werkzeug.utils import secure_filename

app = Flask(__name__)
dropzone = Dropzone(app)
app.config["OUTPUT_DIR"] = config.OUTPUT_IMAGE_DIR

app.config.update(
    SECRET_KEY='QxG07sDClckSHup-DB2Z6w',
    UPLOADED_PATH=config.INPUT_IMAGE_DIR,
    DROPZONE_ALLOWED_FILE_TYPE='image',
    DROPZONE_MAX_FILE_SIZE=3,
    DROPZONE_MAX_FILES=1,
    DROPZONE_UPLOAD_ON_CLICK=True,
    DROPZONE_DEFAULT_MESSAGE="Drop Image files here or click below and upload.",
    DROPZONE_REDIRECT_VIEW="predict"
)

@app.route('/', methods=['POST', 'GET'])
@app.route('/home', methods=['POST', 'GET'])
def home():
    session.clear()
    if request.method == 'POST':
        for key, f in request.files.items():
            if key.startswith('file'):
                image_name = secure_filename(f.filename)
                session["image_name"] = image_name
                f.save(os.path.join(app.config['UPLOADED_PATH'],image_name))
    return render_template("index.html",title="Home")


@app.route("/predict",methods=["GET","POST"])
def predict():
    if request.method == 'GET':
        image_details = {
            "image_name"  : session.get('image_name'),
            "species_name": "",
            "breed_name"  : "",
            "pred_time"   : ""
        }   
        return render_template("predict.html",title="predict",image_details=image_details)
    elif request.method == 'POST':
        image_path = os.path.join(app.config['UPLOADED_PATH'],session.get('image_name'))
        image_details = make_predictions(image_path)
        return render_template("predict.html",title="predict",image_details=image_details)

@app.route('/serve-image/<filename>', methods=['GET'])
def serve_image(filename):
    if filename == session.get('image_name'):
        return send_from_directory(app.config['UPLOADED_PATH'], filename)  
    return send_from_directory(app.config['OUTPUT_DIR'], filename)

@app.route('/class-details', methods=['GET'])
def class_details():
    return render_template("class-info.html",title="classes")

if __name__ == "__main__": 
    app.run(host='0.0.0.0', port=8000)

