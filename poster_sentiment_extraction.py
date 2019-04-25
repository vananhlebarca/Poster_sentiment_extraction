# python 3
import os
import sys
sys.path.insert(0,"/home/anh/caffe-master/python")

import numpy as np
import caffe
import pandas as pd
import urllib.request


mean_file = 'ilsvrc_2012_mean.npy'
    
deploy_path = 'sentiment_deploy.prototxt'
caffemodel_path = 'model/twitter_finetuned_test4_iter_180.caffemodel'


def downloader_tmdb(poster_path,imdb_id):
    image_url = 'http://image.tmdb.org/t/p/w185/' + poster_path
    file_name = 'posters/' + imdb_id
    full_file_name = str(file_name) + '.jpg'
    urllib.request.urlretrieve(image_url,full_file_name)


def main():

	# Load list of posters
	images = pd.read_csv('posters.csv')
	images.head(5)

	# downloading posters
	#for i in range(0,images.shape[0]):
	#	print(i)
	#   	if train.loc[i,'has_poster']==1:
	#   	downloader_tmdb(images.loc[i,'poster_path'],images.loc[i,'imdb_id'])

	# Load network
	net = caffe.Classifier(deploy_path,
                       caffemodel_path,
                       mean=np.load(mean_file).mean(1).mean(1),
                       image_dims=(256, 256),
                       channel_swap=(2, 1, 0),
                       raw_scale=255)
	
	# Load poster
	images['poster_sentiment']=np.nan

	for i in range(0,images.shape[0]):
		print('extracting poster:',i)
		if images.loc[i,'has_poster']==1:
			image_path = '../posters/'+ images.loc[i,'imdb_id']+'.jpg'		
			im = caffe.io.load_image(image_path)
			prediction = net.predict([im]) #, oversample=oversampling)
			images.loc[i,'poster_sentiment']=prediction[0][1]

	images.to_csv('../posters_features.csv',index = False)

	
if __name__=="__main__":
	main()
