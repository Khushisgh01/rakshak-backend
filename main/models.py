from django.db import models


class Camera(models.Model):
    latitude      = models.DecimalField(max_digits=9, decimal_places=6)
    longitude     = models.DecimalField(max_digits=9, decimal_places=6)
    live_feed_url = models.CharField(max_length=500)
    live = models.BooleanField(null=True, blank=True)

class Incident(models.Model):
    latitude      = models.DecimalField(max_digits=9, decimal_places=6)
    longitude     = models.DecimalField(max_digits=9, decimal_places=6)
    incident_type = models.CharField(max_length=500)
    description   = models.CharField(max_length=1000, null=True, blank=True)
    date_created  = models.DateTimeField(null=True)

    def save(self, *args, **kwargs):
        if not self.description:
            self.description = (
                f"An incident of type {self.incident_type} occurred at "
                f"latitude {self.latitude} and longitude: {self.longitude}."
            )
        super().save(*args, **kwargs)
    
class Camera_Incident(models.Model):
    camera        = models.ForeignKey(Camera, on_delete=models.CASCADE)
    incident_type = models.CharField(max_length=500)
    date_created  = models.DateTimeField()
    footage = models.CharField(max_length=1000, null=True)