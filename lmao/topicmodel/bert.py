from .._utils import *
import torch
import gc

if torch.cuda.is_available():
    from cuml.cluster import HDBSCAN
    from cuml.manifold import UMAP
else :
    from hdbscan import HDBSCAN
    from umap import UMAP
    
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

def save_embeddings(df:pd.DataFrame, data_col:str='content', save_path:str=f'{current_dir}/embeddings', bert_model_name:str='all-MiniLM-L6-v2', batch_size:int=1):
    print('Start Embeddings')
    sentence_model = SentenceTransformer(bert_model_name)
    embeddings = sentence_model.encode(df[data_col].to_list(),
                                       show_progress_bar=True,
                                       batch_size=batch_size)
    np.savez_compressed(save_path, embeddings=embeddings)
    gc.collect()
    return save_path+'.npz'

def save_model(topic_model:BERTopic, path_to_save:str, embedding_model:str):
    print(f'Saving BERT model with {embedding_model} embedding model')
    topic_model.save(path_to_save, serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)
    
def load_model(path_to_load:str, embedding_model_name:str='all-MiniLM-L6-v2'):
    # Define embedding model
    embedding_model = SentenceTransformer(embedding_model_name)
    # Load model and add embedding model
    return BERTopic.load(path_to_load, embedding_model=embedding_model)

    
def update_model(df:pd.DataFrame, data_col:str, batch_size = 25000, embeddings_load=None):

    #data preparation
    if embeddings_load:
        print(f'Loading pretrained embedding from path:{embeddings_load}')
        embeddings = np.load(embeddings_load)['embeddings']
    else :
        save_embeddings(df)
        embeddings = np.load(f'{current_dir}/embeddings')['embeddings']

    batch = len(df) // batch_size

    print('Start create a models')

    base_model = BERTopic()

    for i in tqdm(range(batch)):
        umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0)
        hdbscan_model = HDBSCAN(min_samples=10, gen_min_span_tree=True)
        start = i*batch_size
        stop = (i+1)*batch_size
        print('')
        print(f'Started from {start} to {stop} amount:{len(df[start:stop])} docs')
        if i == 0:
            base_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model).fit(df[start:stop][data_col].tolist(), embeddings[start:stop])
        elif i > 0:
            new_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model).fit(df[start:stop][data_col].tolist(), embeddings[start:stop])

            updated_model = BERTopic.merge_models([base_model, new_model])

            # Let's print the newly discover topics
            # nr_new_topics = len(set(updated_model.topics_)) - len(set(base_model.topics_))
            # new_topics = list(updated_model.topic_labels_.values())[-nr_new_topics:]
            # print("The following topics are newly found:")
            # print(f"{new_topics}\n")

            # Update the base model
            base_model = updated_model
            gc.collect()

    rest_start = (batch)*batch_size
    print(f'started from {rest_start} to {len(df)} amount:{len(df[rest_start:])} docs')
    umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0)
    hdbscan_model = HDBSCAN(min_samples=10, gen_min_span_tree=True)
    new_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model).fit(df[rest_start:][data_col].tolist(), embeddings[rest_start:])
    updated_model = BERTopic.merge_models([base_model, new_model])
    gc.collect()
    # Let's print the newly discover topics
    # nr_new_topics = len(set(updated_model.topics_)) - len(set(base_model.topics_))
    # new_topics = list(updated_model.topic_labels_.values())[-nr_new_topics:]
    # print("The following topics are newly found:")
    # print(f"{new_topics}\n")
    
    print('Updating topics with n_gram_range=(1, 2)')
    updated_model.update_topics(df.Content.tolist(), n_gram_range=(1, 2))
    gc.collect()
    return updated_model