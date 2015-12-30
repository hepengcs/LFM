# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import math
import lfm


def generate_data_100k_implicit(k):

    global train, test, release, genres, aux, times
    train = {}
    test = {}
    genres = {}
    release = {}
    aux = {}
    times = {}

    for row in open("ml-100k/u%s.base1" % k, "rU"):
        user, item, rating, time, release_date = row.split('\t')
        user, item, rating, time, release_date = int(user), int(item), int(rating) , int(time), int(release_date)
        train.setdefault(user, {})
        if rating > 3:
            train[user][item] = 1
        else:
            train[user][item] = 0
        release.setdefault(item, 0)
        release[item] = release_date
        times[item] = time

    for row in open("ml-100k/u%s.test" % k, "rU"):
        user, item, rating, time = row.split('\t')
        user, item, rating, time = int(user), int(item), int(rating), int(time)
        test.setdefault(user, {})
        test[user][item] = 1
        times[item] = time

    pre = 0
    for row in open("ml-100k/movie_directors.dat", "rU"):
        temp = row.split('\t')
        item = int(temp[0])
        if item == pre + 1:
            genres.setdefault(item, 0)
            genres[item] = temp[1]
        else:
            for i in range(pre + 1, item + 1):
                genres[i] = 'unknown'
        pre = item

    for row in open("ml-100k/director_counter.dat", "rU"):
        temp = row.split('\t')

        aux.setdefault(temp[0], 1)
        aux[temp[0]] = int(temp[1])
        aux['unknown'] = 1

    global _n, _user_k, _item_k
    _n = 10
    _user_k = 50
    _item_k = 10


def generate_data_100k_explicit(k):

    global train, release, test, genres, aux, times
    train = {}
    release = {}
    genres = {}
    aux = {}
    test = {}
    times = {}
    for row in open("ml-100k/u%s.base1" % k, "rU"):
        user, item, rating, time, release_date = row.split('\t')
        user, item, rating, time, release_date = int(user), int(item), int(rating) , int(time), int(release_date)
        train.setdefault(user, {})
        train[user][item] = rating
        release.setdefault(item, 0)
        release[item] = release_date
        times[item] = time


    pre = 0
    for row in open("ml-100k/movie_directors.dat", "rU"):
        temp = row.split('\t')
        item = int(temp[0])
        if item == pre + 1:
            genres.setdefault(item, 0)
            genres[item] = temp[1]
        else:
            for i in range(pre + 1, item + 1):
                genres[i] = 'unknown'
        pre = item


    for row in open("ml-100k/director_counter.dat", "rU"):
        temp = row.split('\t')

        aux.setdefault(temp[0], 1)
        aux[temp[0]] = int(temp[1])
        aux['unknown'] = 1


    for row in open("ml-100k/u%s.test" % k, "rU"):
        user, item, rating, time = row.split('\t')
        user, item, rating, time = int(user), int(item), int(rating), int(time)
        test.setdefault(user, {})
        test[user][item] = rating
        times[item] = time


def generate_matrix(implicit, g1, g2, w1, w2, w3, f, l):

    lfm.factorization(train, release, genres, aux, times, bias=True, svd=True, svd_pp=False, steps=25, gamma1=g1, gamma2=g2, w1=w1, w2=w2, w3=w3, k=f, Lambda=l)  # explicit
    # lfm.factorization(train, release, genres, aux, times,bias=True, svd=True, svd_pp=False, steps=25, gamma1=g1, gamma2=g2, w1=w1,w2=w2,w3=w3,slow_rate=0.9, Lambda=0.01,
     #                  ratio=7)  # implicit



def get_recommendation_explicit(user):

     return lfm.recommend_explicit(user)


def get_recommendation_implicit(user):

    return lfm.recommend_implicit(user, _n)





def recall():

    hit = 0
    count = 0
    for user in train.iterkeys():
        tu = test.get(user, {})
        rank = get_recommendation_implicit(user)
        for item, pui in rank:
            if item in tu:
                hit += 1
        count += len(tu)
    return hit / count


def precision():
    hit = 0
    count = 0
    for user in train.iterkeys():
        tu = test.get(user, {})
        rank = get_recommendation_implicit(user)
        for item, pui in rank:
            if item in tu:
                hit += 1
        count += len(rank)
    return hit / count


def coverage():

    recommend_items = set()
    all_items = set()
    for user in train.iterkeys():
        for item in train[user].iterkeys():
            all_items.add(item)
        rank = get_recommendation_implicit(user)
        for item, pui in rank:
            recommend_items.add(item)
    return len(recommend_items) / len(all_items)


def popularity():

    item_popularity = {}
    for items in train.itervalues():
        for item in items.iterkeys():
            item_popularity.setdefault(item, 0)
            item_popularity[item] += 1
    popularity_sum = 0
    count = 0
    for user in train.iterkeys():
        rank = get_recommendation_implicit(user)
        for item, pui in rank:
            popularity_sum += math.log(1 + item_popularity[item])
        count += len(rank)
    return popularity_sum / count


def RMSE():

    rmse_sum = 0
    hit = 0
    for user in train.iterkeys():
        tu = test.get(user, {})
        rank = get_recommendation_implicit(user)
        for item, pui in rank:
            if item in tu:
                rmse_sum += (tu[item] - pui) ** 2
                hit += 1
    return math.sqrt(rmse_sum / hit)


def MAE():

    mae_sum = 0
    hit = 0
    for user in train.iterkeys():
        tu = test.get(user, {})
        rank = get_recommendation_implicit(user)
        for item, pui in rank:
            if item in tu:
                mae_sum += abs(tu[item] - pui)
                hit += 1
    return mae_sum / hit



def evaluate_explicit():
    hit = 0
    rmse_sum = 0
    mae_sum = 0
    for user in train.iterkeys():
        tu = test.get(user, {})
        rank = get_recommendation_explicit(user)
        for item, pui in rank:
            if item in tu:
                hit += 1
                rmse_sum += (tu[item] - pui) ** 2
                mae_sum += abs(tu[item] - pui)
    rmse_value = math.sqrt(rmse_sum / hit)
    mae_value = mae_sum / hit
    return rmse_value, mae_value


def evaluate_implicit():
    item_popularity = {}
    for items in train.itervalues():
        for item in items.iterkeys():
            item_popularity.setdefault(item, 0)
            item_popularity[item] += 1
    hit = 0
    test_count = 0
    recommend_count = 0
    recommend_items = set()
    all_items = set()
    popularity_sum = 0
    for user in train.iterkeys():
        tu = test.get(user, {})
        rank = get_recommendation_implicit(user)
        for item, pui in rank:
            if item in tu:
                hit += 1
            recommend_items.add(item)
            popularity_sum += math.log(1 + item_popularity[item])
        test_count += len(tu)
        recommend_count += len(rank)
        for item in train[user].iterkeys():
            all_items.add(item)
    recall_value = hit / test_count
    precision_value = hit / recommend_count
    coverage_value = len(recommend_items) / len(all_items)
    popularity_value = popularity_sum / recommend_count
    return recall_value, precision_value, coverage_value, popularity_value
