import os
from dotenv import load_dotenv

load_dotenv()

DATA_SOURCE = os.getenv("DATA_SOURCE", "local")

AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_KEY_PROFILES = os.getenv("S3_KEY_PROFILES")
S3_KEY_DESCRIPTORS = os.getenv("S3_KEY_DESCRIPTORS")