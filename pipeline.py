from AIS_final_model_pipeline.ipynb import train_and_evalute_model_pipeline
from AIS_final_model_pipeline.ipynb import benchmark_dict
from surprise.prediction_algorithms import NMF
from surprise.prediction_algorithms import SVD
from surprise.prediction_algorithms import SVD++

my_model, metrics_dict = train_and_evalute_model_pipeline(NMF)
metrics_dict

my_model, metrics_dict = train_and_evalute_model_pipeline(SVD)
metrics_dict

my_model, metrics_dict = train_and_evalute_model_pipeline(SVD++)
metrics_dict

benchmark_dict = {}

model_kwargs = {'user_based': True, 'name': 'cosine'}
nmf, metrics_dict = train_and_evalute_model_pipeline(NMF, model_kwargs)
benchmark_dict['NMF user based cosine'] = metrics_dict

model_kwargs = {'user_based': True, 'name': 'pearson'}
nmf, metrics_dict = train_and_evalute_model_pipeline(NMF, model_kwargs)
benchmark_dict['NMF user based pearson'] = metrics_dict

model_kwargs = {'user_based': False, 'name': 'cosine'}
svd, metrics_dict = train_and_evalute_model_pipeline(SVD, model_kwargs)
benchmark_dict['SVD item based cosine'] = metrics_dict

model_kwargs = {'user_based': False, 'name': 'pearson'}
svd++, metrics_dict = train_and_evalute_model_pipeline(SVD, model_kwargs)
benchmark_dict['SVD item based pearson'] = metrics_dict

model_kwargs = {'user_based': False, 'name': 'cosine'}
svd, metrics_dict = train_and_evalute_model_pipeline(SVD++, model_kwargs)
benchmark_dict['SVD++ item based cosine'] = metrics_dict

model_kwargs = {'user_based': False, 'name': 'pearson'}
svd++, metrics_dict = train_and_evalute_model_pipeline(SVD++, model_kwargs)
benchmark_dict['SVD++ item based pearson'] = metrics_dict