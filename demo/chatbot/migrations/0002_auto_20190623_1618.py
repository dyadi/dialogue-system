# Generated by Django 2.2.2 on 2019-06-23 16:18

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('chatbot', '0001_initial'),
    ]

    operations = [
        migrations.DeleteModel(
            name='Movie',
        ),
        migrations.DeleteModel(
            name='Theater',
        ),
        migrations.DeleteModel(
            name='Timetable',
        ),
    ]
