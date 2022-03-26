from operator import mod
from django.db import models

# Create your models here.
class Profile(models.Model):
    name = models.CharField(max_length=255)
    img = models.ImageField(upload_to="img")

    def __str__(self):
        return self.name


class Attendance(models.Model):
    e_id = models.ForeignKey(Profile, on_delete=models.CASCADE, default="")
    name = models.CharField(max_length=255)
    date = models.DateTimeField()