from rest_framework.serializers import Serializer,FileField

class AudioSerializer(Serializer):
    audio = FileField()