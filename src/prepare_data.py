from operator import add
import numpy as np
import pandas as pd
import h5py
import os
from numpy import random


def get_overview(city_list, path):
    no_of_votes_tmp = []
    for city in city_list:
        name_tmp = path + city + ".h5"
        try:
            h5_tmp = h5py.File(name_tmp, 'r')
            no_of_votes_tmp.append(np.array(h5_tmp['label']).shape[1])
        except FileNotFoundError:
            print(f"File not found: {name_tmp}")
            no_of_votes_tmp.append(None)  # Append None or any other value indicating missing file
        except Exception as e:
            print(f"An error occurred: {e}")
            no_of_votes_tmp.append(None)  # Handle other potential errors

    return no_of_votes_tmp


def concatenate_cities(cities_list, path):
    concatenated_mat = np.array([])
    for city in cities_list:
        name_tmp = path + city + ".h5"
        h5_tmp = h5py.File(name_tmp,'r')
        if concatenated_mat.size == 0:
            concatenated_mat = np.array(h5_tmp['label'])
        else:
            concatenated_mat = np.vstack((concatenated_mat, np.array(h5_tmp['label'])))
    return(concatenated_mat)


def process_city(city, path):
    name_tmp = path + city + ".h5"
    h5_tmp = h5py.File(name_tmp,'r')
    votes_mat = pd.DataFrame(np.array(h5_tmp['label']))
    city_mat = np.array([city] * len(h5_tmp['label']))
    votes_mat['City'] = city_mat
    return votes_mat


def process_geo(city, path):
    name_tmp = path + city + ".h5"
    h5_tmp = h5py.File(name_tmp,'r')
    geo_mat = pd.DataFrame(np.array(h5_tmp['img_coord']))

    # relevant columns: 1 and 2 in UTM
    geo_mat = geo_mat[[0,1]]

    # get city column
    city_mat = np.array([city] * len(h5_tmp['img_coord']))
    geo_mat['City'] = city_mat

    return geo_mat


def to_one_hot(vote_mat, labels):
    one_hot_encoded_mat = list()
    for i in range(len(vote_mat)):
        one_hot_encoded = list()
        vote_vec = vote_mat[i,:]
        for value in vote_vec:
            one = [0 for _ in range(len(labels))]
            one[value-1] = 1
            if one_hot_encoded:
                one_hot_encoded = list(map(add, one_hot_encoded, one))
            else: # initialize one_hot_encoded if list is yet empty
                one_hot_encoded = one
        if i == 0:
            one_hot_encoded_mat = np.asarray(one_hot_encoded)
        else:
            one_hot_encoded_mat = np.vstack((one_hot_encoded_mat, np.asarray(one_hot_encoded)))
    return(one_hot_encoded_mat)


def set_seeds(seed_value = 56):
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)


def get_dataframes(city_list, path):
    city_frames = [process_city(city, path) for city in city_list]
    all_cities_named = pd.concat(city_frames)
    all_cities_named = all_cities_named[np.all(all_cities_named != 0, axis=1)]

    all_cities = concatenate_cities(city_list, path).astype(int)
    all_cities = all_cities[np.all(all_cities != 0, axis=1)]

    cities_one_hot_named = pd.DataFrame(to_one_hot(all_cities, labels=np.arange(1,18)))
    cities_one_hot_named['City'] = np.array(all_cities_named['City'])

    cities_one_hot = np.array(cities_one_hot_named.drop('City', axis=1))

    return all_cities_named, all_cities, cities_one_hot_named, cities_one_hot


def get_dataframes_votes(city_list, path):
    city_frames = [process_city(city, path) for city in city_list]
    all_cities_named = pd.concat(city_frames)
    all_cities_named = all_cities_named[np.all(all_cities_named != 0, axis=1)]

    all_cities = concatenate_cities(city_list, path).astype(int)
    all_cities = all_cities[np.all(all_cities != 0, axis=1)]

    return all_cities_named, all_cities

