# Generated by Django 4.0.3 on 2022-03-12 13:51

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('face_detect', '0002_attendance'),
    ]

    operations = [
        migrations.AddField(
            model_name='attendance',
            name='e_id',
            field=models.ForeignKey(default='', on_delete=django.db.models.deletion.CASCADE, to='face_detect.profile'),
        ),
    ]