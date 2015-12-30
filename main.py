# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import helper

import numpy as np
import pandas as pd
import csv
import os


def test100k_explicit():
    test_count = 1
    evaluation_base = 2
    for f in [30]:
        for weight1 in [1]:
            for weight2 in [0]:
                for weight3 in [1]:
                    for gamma1 in [0.04]:
                        for gamma2 in [0.04]:
                            for lr in [0.1]:
                                ans = [0] * evaluation_base
                                for k in xrange(1, test_count + 1):
                                    helper.generate_data_100k_explicit(k)
                                    helper.generate_matrix(implicit=False, g1=gamma1, g2=gamma2, w1=weight1, w2=weight2, w3=weight3, f=f, l=lr)
                                    b = helper.evaluate_explicit()
                                    for x in xrange(evaluation_base):
                                        ans[x] += b[x]
                                for x in xrange(evaluation_base):
                                    ans[x] /= test_count
                                print weight1, weight2, weight3, ans


def test100k_implicit():
    test_count = 1
    evaluation_base = 4
    for gamma1 in [0.04]:
        for gamma2 in [0.04]:
            ans = [0] * evaluation_base
            for k in xrange(1, test_count + 1):
                helper.generate_data_100k_implicit(k)
                helper.generate_matrix(implicit=True, g1=gamma1, g2=gamma2, w1=1, w2=0, w3=1, f=25, l=0.1)
                b = helper.evaluate_implicit()
                for x in xrange(evaluation_base):
                    ans[x] += b[x]
            for x in xrange(evaluation_base):
                ans[x] /= test_count
            print ans



def merge_data():
    i_cols = ['movie_id', 'release_date']
    movies = pd.read_csv('./ml-100k/u.item', sep='|', usecols=[0, 2], names=i_cols)
    movies['release_date'] = movies['release_date'].str[6:] + movies['release_date'].str[3:5] + movies['release_date'].str[0:2] 
    movies = movies.sort('release_date')


    r_cols = ['user_id', 'movie_id', 'ratings', 'times']
    ratings = pd.read_csv('./ml-100k/u1.base', sep='\t', names=r_cols, index_col=False)
    result = pd.merge(ratings, movies, on='movie_id')
    result.to_csv('./ml-100k/u1.base1', index=False, sep='\t', header=False)


    d_cols = ['movie_id', 'director']
    directors = pd.read_csv('./ml-100k/movie_directors.dat', sep='\t', names=d_cols, index_col=False, usecols=[0, 1])

    directors = pd.merge(ratings, directors, on='movie_id')
    directors = directors.groupby('director').size()
    directors.to_csv('./ml-100k/director_counter.dat', index=True, sep='\t', header=False)



if __name__ == '__main__':
    merge_data()
    test100k_explicit()
    # test100k_implicit()
