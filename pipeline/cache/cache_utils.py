import numpy as np
import pandas as pd
import torch
from functools import partial
from PIL import Image
from typing import Union
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

import clip_embed
from qdrant_client import QdrantClient, models

from upload_s3 import S3Uploader
from torchvision.transforms import functional

# Initialize s3 uploader
su = S3Uploader()
BUCKET_NAME = ''
client = QdrantClient(url="")

## make a vector database call
def search_nearest_cache_vdb(prompt_i=''):
    query_vector = clip_embed.embed_text(prompt_i)[0].tolist()
    hits = client.search(
        collection_name="]",
        query_vector=query_vector,
        limit=1,  # Return top 5 results
        with_payload=True,  # Return metadata
    )

    return hits


## retrieve noise from EFS or S3
def retrieve_noise_vdb(hits=None, K_1=None):
    assert K_1 is not None and hits is not None
    top_1_intermediate_noise = []
    if K_1 == 0:
        top_1_intermediate_noise.append(None)
    else:
        top_1_intermediate_noise.append(
            su.get_tensor(bucket=BUCKET_NAME, key=hits[0].payload[f'x0_step_{K_1}_seed_0']))

    return top_1_intermediate_noise


## for trying out different combinations of generation quality, hit rate, and compute savings,
## please try to change the scores in if-else blocks
### heuristics for deciding K
def heuristics_K(similarity_score):
    if similarity_score > 0.95:
        k = 25
    elif similarity_score > 0.9:
        k = 20
    elif similarity_score > 0.85:
        k = 15
    elif similarity_score > 0.75:
        k = 10
    elif similarity_score > 0.65:
        k = 5
    else:
        k = 0
    return k

## pipeline for generation using cache and instantaneous cache maintenance
def generate_image_using_cache(prompt_i, default_k=0, seeds=[0, 1]):
    hits = cache_search.search_nearest_cache_vdb(prompt_i)
    similarity_score_top_1 = hits[0].score

    images_list = []

    K_1 = heuristics_K(similarity_score_top_1)

    if ((default_k > 0) & (default_k < 50) & ((default_k - 1) % 4 == 0)):

        [top_1_intermediate, top_2_intermediate] = cache_search.retrieve_noise_vdb(hits=hits, K_1=default_k - 1)
        [top_1_intermediate, top_2_intermediate] = torch.from_numpy(np.array([top_1_intermediate, top_2_intermediate]))
        app.config.iteration_parameters.interval_points = [default_k / 50, 1]
        app.config.seeds = [seeds[0]]
        Image_seed_0_top_1 = app(text_prompt=prompt_i, initial_image=top_1_intermediate).images
        images_list = [Image_seed_0_top_1]

        return images_list

    [top_1_intermediate] = cache_search.retrieve_noise_vdb(hits=hits, K_1=K_1 - 1)

    [top_1_intermediate] = torch.from_numpy(np.array([top_1_intermediate]))

    app.config.return_samples_only = False
    app.config.iteration_parameters.add_noise_to_initial_image = True

    if K_1 != 0:
        app.config.iteration_parameters.interval_points = [K_1 / 50, 1]
        app.config.seeds = [seeds[0]]
        Image_seed_0_top_1 = app(text_prompt=prompt_i, initial_image=top_1_intermediate).images

    else:
        app.config.iteration_parameters.interval_points = [0, 1]
        app.config.seeds = [seeds[0]]
        Image_seed_0_top_1 = app(text_prompt=prompt_i).images

    images_list = [Image_seed_0_top_1]

    return images_list


## sample code for LCBFU cache
MAX_LIMIT = 1500000

def retrieve_cache_and_maintain(query_prompt, query_embed, index):
    global df_cache_small
    idx_access = np.argmax(cosine_similarity(np.array(df_cache_small.embed.to_list()), [query_embed]))
    print("idx_access ", idx_access)
    K = heuristics_K(np.max(cosine_similarity(np.array(df_cache_small.embed.to_list()), [query_embed])))
    print(np.max(cosine_similarity(np.array(df_cache_small.embed.to_list()), [query_embed])))
    if K > 0:
        cache_item = df_cache_small.loc[int(int(idx_access / 5) * 5 + (K / 5 - 1))]
        assert K == cache_item.K
        print("PROMPT = ", cache_item.prompt, " ; K = ", cache_item.K)
        noise_img_key = cache_item[f'x0_step_{K * 2}_seed_0']
        img_key = cache_item[f'image_seed_0']
        img_orig = su.get_image(BUCKET_NAME, img_key)
        img_orig.resize((256, 256)).show()
        print(noise_img_key)
        noise_img = torch.from_numpy(
            np.array([su.get_tensor(bucket=BUCKET_NAME, key=noise_img_key)]))
        noise2img(app, np.array([su.get_tensor(bucket=BUCKET_NAME, key=noise_img_key)])).resize((256, 256)).show()

        image_original_without_cache = merged_df[merged_df.prompt == query_prompt].image_seed_0.to_list()[0]
        su.get_image(BUCKET_NAME, image_original_without_cache).resize((256, 256)).show()

        app.config.return_samples_only = False
        app.config.iteration_parameters.add_noise_to_initial_image = True
        app.config.iteration_parameters.num_iters = 100
        app.config.iteration_parameters.interval_points = [(K * 2) / 100, 1]
        app.config.seeds = [0]

        Image_seed_0_top_1 = app(text_prompt=query_prompt, initial_image=noise_img).images

        Image_seed_0_top_1.resize((256, 256)).show()

        df_cache_small.loc[int(int(idx_access / 5) * 5 + (K / 5 - 1)), 'freq'] += 1
        if df_cache_small.loc[int(int(idx_access / 5) * 5 + (K / 5 - 1)), 'access_times'] is None:
            df_cache_small.loc[int(int(idx_access / 5) * 5 + (K / 5 - 1)), 'access_times'] = [5000 + 1]
        else:
            df_cache_small.loc[int(int(idx_access / 5) * 5 + (K / 5 - 1)), 'access_times'].append(5000 + 1)

    else:
        print("Not Found")
        if len(df_cache_small) == MAX_LIMIT:
            print("making short - Found")
            df_cache_small['freq_K_rank'] = df_cache_small.apply(lambda x: x['freq'] * (x['K'] / 50), axis=1)
            df_cache_small = df_cache_small.sort_values('freq_K_rank')[5:]  # Proposed

        df_cache_small = pd.concat([df_cache_small, merged_df.loc[merged_df['index'] == index]], ignore_index=True)

# Unit test
def test():
    for query_i in range(0, 1000000):
        print("Query_prompt ", merged_df.loc[query_i, "prompt"])
        index_query = merged_df.loc[query_i, 'index']
        query_embed = merged_df.loc[query_i, 'embed']
        query_prompt = merged_df.loc[query_i, "prompt"]
        generate_using_cache(query_prompt, query_embed, index_query)
    return


if __name__ == "__main__":
    # You can call your functions here or run the unit test
    # test()
    pass
