from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.views import APIView
import pickle
from statistics import mode
import librosa
import soundfile
import io
import numpy as np
from .serializers import AudioSerializer
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

class PredictAudioEmotion(APIView):
    serializer_class = AudioSerializer    
    def post(self, request, format=None):
        try:
            model = pickle.load(open('finalized_model.sav', 'rb'))
            audio_data = request.FILES.get('audio')
            path = default_storage.save(
                str(settings.BASE_DIR) + '/audio'+'/recording.wav', ContentFile(audio_data.read()))
            file = path
            feature = extract_feature(file,mfcc=True,chroma=True,mel=True)
            y_pre=model.predict([feature])
            return Response({'emotion':y_pre})
        except Exception as e:
            return Response({"error":str(e)},status=500)
