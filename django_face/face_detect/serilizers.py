
from rest_framework.serializers import ModelSerializer
from face_detect.models import Profile,Attendance


class profileserializer(ModelSerializer):

    class Meta:
        model  = Profile
        fields = '__all__'


class Attenserializer(ModelSerializer):

    class Meta:
        model  = Attendance
        fields = '__all__'

    def create(self, data):
        return Attendance.objects.create(**data)