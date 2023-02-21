from codecs import open
import os
import uuid
import boto3



class Model(object):
  def __init__(self):
  	self.nothing = 0

  def save_image(self, drawn_letter, image):
  	filename = 'letter' + str(drawn_letter) + '__' + str(uuid.uuid1()) + '.jpg'
  	with open('tmp/' + 'draw_image.jpg', 'wb') as f:
  	    f.write(image)

  	AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
  	AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
  	s3 = boto3.resource(service_name='s3', region_name='eu-central-1', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
  	s3.Bucket('hebrewlettersawsbucket').upload_file(Filename='tmp/' + 'draw_image.jpg', Key=f'{drawn_letter}/' + filename)

  	return ('Image saved successfully with the name {0}'.format(filename))