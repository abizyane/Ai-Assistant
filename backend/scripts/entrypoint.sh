#!/bin/sh

cd /app
python manage.py makemigrations chat notification astropong game
python manage.py migrate

exec python manage.py runserver 0.0.0.0:8000