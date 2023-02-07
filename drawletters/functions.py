__author__ = 'Artgor'
from codecs import open
import os
import uuid
import boto3



class Model(object):
  def __init__(self):
  	self.nothing = 0

  def save_image(self, drawn_digit, image):
  	filename = 'digit' + str(drawn_digit) + '__' + str(uuid.uuid1()) + '.jpg'
  	with open('tmp/' + 'draw_image.jpg', 'wb') as f:
  	    f.write(image)

  	REGION_HOST = 's3-accesspoint.eu-central-1.amazonaws.com'
  	AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
  	AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
  	s3 = boto3.resource(service_name='s3', region_name='eu-central-1', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
  	s3.Bucket('hebrewlettersawsbucket').upload_file(Filename='tmp/' + 'draw_image.jpg', Key=f'{drawn_digit}/' + filename)

#   	conn = S3Connection(AWS_ACCESS_KEY_ID,
#                                 AWS_SECRET_ACCESS_KEY,
#                                         host=REGION_HOST)

#   	bucket = conn.get_bucket("hebrewlettersawsbucket")

#   	k = Key(bucket)
#   	fn = 'tmp/' + filename
#   	k.key = filename
#   	k.set_contents_from_filename(fn)

  	return ('Image saved successfully with the name {0}'.format(filename))