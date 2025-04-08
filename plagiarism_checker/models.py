from django.db import models
class PlagiarismCheck(models.Model):
    text = models.TextField()
    similarity_score = models.FloatField()
    checked_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Checked on {self.checked_at}"
# Create your models here.
