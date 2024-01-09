from flask import Flask, jsonify, request, render_template
import numpy as np
import librosa
import pickle

app = Flask(__name__)
app.static_folder = 'static'


@app.route('/')
def index():
    return render_template('index.html')


def extract_feature(file_stream, mfcc):
    X, sample_rate = librosa.load(file_stream, res_type='kaiser_fast')
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(
            y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    return result


with open('Emotion_Voice_Detection_Model.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/upload', methods=['POST'])
def upload_file():
    # Get the uploaded file
    audio_file = request.files.get('audio_file')

    # Chec k if the file is empty
    if not audio_file:
        return jsonify({"message": "No file provided"}), 400

    new_feature = extract_feature(audio_file, mfcc=True)
    ans = []
    ans.append(new_feature)
    ans = np.array(ans)
    prediction = model.predict(ans)

    return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
