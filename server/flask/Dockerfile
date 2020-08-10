# Pull tensorflow image with Python3
FROM tensorflow/tensorflow:2.1.0-py3
WORKDIR /app
ADD . /app
RUN apt-get update && apt-get install -y libsndfile1 libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg
RUN pip install -r requirements.txt
CMD ["uwsgi", "app.ini"]