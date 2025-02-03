# Generated by Django 5.1.5 on 2025-01-24 08:06

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Booking",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=100)),
                ("stay_period", models.CharField(max_length=50)),
                ("guests", models.IntegerField()),
                ("breakfast", models.BooleanField(default=False)),
                ("payment_method", models.CharField(max_length=50)),
            ],
        ),
        migrations.CreateModel(
            name="ChatbotResponse",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("question", models.CharField(max_length=255)),
                ("answer", models.TextField()),
            ],
        ),
    ]
