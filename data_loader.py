from io import BytesIO
import pandas as pd
import boto3
import logging
from config import (DATA_SOURCE, S3_BUCKET, S3_KEY_PROFILES, S3_KEY_DESCRIPTORS,
                    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)


logger = logging.getLogger(__name__)


def load_data(s3_key=S3_KEY_PROFILES):
    if DATA_SOURCE == "s3":
        if not S3_BUCKET or not s3_key:
            raise ValueError("S3_BUCKET and S3_KEY must be set when DATA_SOURCE=s3")

        s3 = boto3.client(
            "s3",
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
        )
        logger.info("Loading data from S3")

        if s3_key == S3_KEY_PROFILES:
            profiles = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY_PROFILES)

            profiles_df = pd.read_csv(BytesIO(profiles["Body"].read()))
            logger.info("Profile data loaded successfully into dataframe")
            return profiles_df
        elif s3_key == S3_KEY_DESCRIPTORS:
            descriptors_object = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY_DESCRIPTORS)

            descriptors = BytesIO(descriptors_object["Body"].read())
            mouthfeel = pd.read_excel(descriptors, sheet_name='Mouthfeel', engine='openpyxl')
            taste = pd.read_excel(descriptors, sheet_name='Taste', engine='openpyxl')
            flavor = pd.read_excel(descriptors, sheet_name='Flavor And Aroma', engine='openpyxl')

            logger.info("Descriptor data loaded successfully into dataframe")
            return mouthfeel, taste, flavor
        logger.info("Data was not loaded form S3...")
    return None
