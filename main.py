# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START app]
from datetime import datetime
import logging
import os
import numpy as np
import scipy
import string
import random
from clarifai.rest import ClarifaiApp

import scipy.misc

from flask import Flask, redirect, render_template, request, url_for

from google.cloud import datastore
from google.cloud import storage
from google.cloud import texttospeech


CLOUD_STORAGE_BUCKET = os.environ.get('CLOUD_STORAGE_BUCKET')


app = Flask(__name__)


@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/results', methods=['GET'])
def results():
    # Create a Cloud Datastore client.
    datastore_client = datastore.Client()

    # Use the Cloud Datastore client to fetch information from Datastore about
    # each photo.
    query = datastore_client.query(kind='Images')
    query.order = ['-timestamp']
    image_entity = list(query.fetch())[0]

    query1 = datastore_client.query(kind='Audio')
    query1.order = ['-timestamp']
    audio_entity = list(query1.fetch())[0]
    # Return a Jinja2 HTML template and pass in image_entities as a parameter.
    return render_template('results.html', image_entity=image_entity, audio_entity=audio_entity)

@app.route('/upload_photo', methods=['GET', 'POST'])
def upload_photo():
    photo = request.files['file']

    name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    # Create a Cloud Storage client.
    storage_client = storage.Client()

    # Get the bucket that the file will be uploaded to.
    bucket = storage_client.get_bucket(CLOUD_STORAGE_BUCKET)

    # Create a new blob and upload the file's content.
    blob = bucket.blob(name+'.jpg')
    blob.upload_from_string(photo.read(), content_type=photo.content_type)

    # Make the blob publicly viewable.
    blob.make_public()

    audioname = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

    tts_client = texttospeech.TextToSpeechClient()

    clarifaiClient = ClarifaiApp()

    model = clarifaiClient.models.get('foodcam')
    out = model.predict_by_url(blob.public_url)
    best = str(out['outputs'][0]['data']['concepts'][0]['id'])

    intext = texttospeech.types.SynthesisInput(text=best)

    voice = texttospeech.types.VoiceSelectionParams(
        language_code='en-US',
        ssml_gender=texttospeech.enums.SsmlVoiceGender.NEUTRAL)

    # Select the type of audio file you want returned
    audio_config = texttospeech.types.AudioConfig(
        audio_encoding=texttospeech.enums.AudioEncoding.MP3)

    outspeech = tts_client.synthesize_speech(intext, voice, audio_config)

    with open('output.mp3', 'wb') as out:
        # Write the response to the output file.
        out.write(outspeech.audio_content)

    speechblob = bucket.blob(audioname+'.mp3')
    speechblob.upload_from_filename('output.mp3')

    speechblob.make_public()

    # Create a Cloud Datastore client.
    datastore_client = datastore.Client()

    # Fetch the current date / time.
    current_time = datetime.now()
    current_datetime = current_time.timestamp()

    # The kind for the new entity.
    kind = 'Images'

    # The name/ID for the new entity.
    name = blob.name

    # Create the Cloud Datastore key for the new entity.
    key = datastore_client.key(kind, name)

    # Construct the new entity using the key. Set dictionary values for entity
    # keys blob_name, storage_public_url, timestamp, and joy.
    entity = datastore.Entity(key)
    entity['blob_name'] = blob.name
    entity['image_public_url'] = blob.public_url
    entity['timestamp'] = current_datetime
    entity['best'] = best

    # Save the new entity to Datastore.
    datastore_client.put(entity)

    kind1 = 'Audio'

    # The name/ID for the new entity.
    name1 = speechblob.name

    # Create the Cloud Datastore key for the new entity.
    key1 = datastore_client.key(kind1, name1)

    # Construct the new entity using the key. Set dictionary values for entity
    # keys blob_name, storage_public_url, timestamp, and joy.
    entity1 = datastore.Entity(key1)
    entity1['blob_name'] = speechblob.name
    entity1['speech_public_url'] = speechblob.public_url
    entity1['timestamp'] = current_datetime

    # Save the new entity to Datastore.
    datastore_client.put(entity1)

    # Redirect to the home page.
    return redirect('/results')


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END app]
