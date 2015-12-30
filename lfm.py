# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import heapq
import operator

import numpy
import datetime
import time


def factorization(train, r, g, au, ts, bias=False, svd=True, svd_pp=False, steps=25, gamma=0.04, gamma1=0.04 , gamma2=0.04,
    slow_rate=0.93, Lambda=0.1, k=30, k1=30, k2=30, ratio=None, seed=0, w1=1, w2=1, w3=1, pop=False):

    global release, genres, aux
    release = r
    genres = g
    aux = au

    global _user_items
    _user_items = train
    numpy.random.seed(seed)
    global _bias, _svd, _svd_pp, _k, _k1, _k2, _weight1, _weight2, _weight3, _times
    _bias = bias
    _svd = svd
    _svd_pp = svd_pp
    _k = k
    _k1 = k1
    _k2 = k2
    _weight1 = w1
    _weight2 = w2
    _weight3 = w3
    _times = ts
    global _bu, _bi, _pu, _pu1, _pu2, _pi, _qi, _qt, _qi1, _qi2, _qi3, _z, _movie_list, _movie_set, _avr, _tot
    global _bt, _bd
    _bt = {}
    _bu = {}
    _bi = {}
    _pu = {}
    _pu1 = {}
    _pu2 = {}
    _qi = {}
    _qi1 = {}
    _qi2 = {}
    _qi3 = {}
    _pi = {}
    _qt = {}
    _z = {}
    _bd = {}
    sqrt_item_len = {}
    _movie_list = []
    _avr = 0
    _tot = 0
    y = {}

    for i in range(36500):
        _qt.setdefault(i, numpy.random.random((_k2, 1)) / numpy.sqrt(_k2))

    for user, items in _user_items.iteritems():

        if _bias:
            _bu.setdefault(user, 0)
        if _svd:
            _pu.setdefault(user, numpy.random.random((_k, 1)) / numpy.sqrt(_k) * w1)
            _pu1.setdefault(user, numpy.random.random((_k1, 1)) / numpy.sqrt(_k1) * w2)
            _pu2.setdefault(user, numpy.random.random((_k2, 1)) / numpy.sqrt(_k2) * w3)


        if _svd_pp:
            sqrt_item_len.setdefault(user, numpy.sqrt(len(items)))

        for item, rating in items.iteritems():
            _movie_list.append(item)
            if _bias:
                _bi.setdefault(item, 0)
            if _svd:
                _qi.setdefault(item, numpy.random.random((_k, 1)) / numpy.sqrt(_k) * w1)
                _qi1.setdefault(release[item], numpy.random.random((_k1, 1)) / numpy.sqrt(_k1) * w2)
                genres.setdefault(item, 0)
                _qi2.setdefault(genres[item], numpy.random.random((_k2, 1)) / numpy.sqrt(_k2) * w3)
                _pi.setdefault(item, numpy.random.random((_k2, 1)) / numpy.sqrt(_k2))
                _bd.setdefault(genres[item], 0)

            if _svd_pp:
                y.setdefault(item, numpy.zeros((_k, 1)))
            _avr += rating
            _tot += 1
 


    _movie_set = set(_movie_list)
    _avr /= _tot
    for step in xrange(steps):
        rmse_sum = 0
        mae_sum = 0
        for user, items in _user_items.iteritems():
            samples = items if not ratio else __random_negative_sample(items, ratio)

            for item, rating in samples.iteritems():
                eui = rating - __predict(user, item)
                rmse_sum += eui ** 2
                mae_sum += abs(eui)
                if _bias:
                    _bu[user] += gamma * (eui - Lambda * _bu[user])
                    _bi[item] += gamma * (eui - Lambda * _bi[item])

                if _svd:
                    release_date = release[item]
                    f2 = genres[item]
                    timestamp = _times[item]


                    if _weight1 != 0:
                        _pu[user], _qi[item] = _pu[user] + gamma * (eui * _qi[item] - Lambda * _pu[user]), _qi[
                            item] + gamma * (eui * (_pu[user] + _z[user] if _svd_pp else _pu[user]) - Lambda * _qi[item])
                    
                    if _weight2 != 0:
                        _pu1[user], _qi1[release_date] = _pu1[user] + gamma1 * (eui * _qi1[release_date] - Lambda * _pu1[user]), _qi1[
                            release_date] + gamma1 * (eui * (_pu1[user] + _z[user] if _svd_pp else _pu1[user]) - Lambda * _qi1[release_date])

                    if _weight3 != 0:
                        _pu2[user], _qi2[f2] = _pu2[user] + gamma2 * (eui / aux[f2] * _qi2[f2] - Lambda * _pu2[user]), _qi2[
                            f2] + gamma2 / aux[f2] * (eui * (_pu2[user] + _z[user] if _svd_pp else _pu2[user]) - Lambda * _qi2[f2])
                        
                        # _bd[f2] += gamma * (eui - Lambda * _bd[f2])
                       
                    if pop:
                        days = compute_days(timestamp, release_date)

                        _pi[item], _qt[days] = _pi[item] + gamma2 * (eui * _qt[days] - Lambda * _pi[item]), _qt[
                            days] + gamma2 * (eui * (_pi[item] + _z[user] if _svd_pp else _pi[item]) - Lambda * _qt[days])
                       
        gamma *= slow_rate
        gamma1 *= slow_rate
        gamma2 *= slow_rate
        print "step: %s, rmse: %s, mae: %s" % (step + 1, numpy.sqrt(rmse_sum / _tot), mae_sum / _tot)


def __random_negative_sample(items, ratio=1):

    ret = {}
    for item in items.iterkeys():
        ret[item] = 1
        
    n = 0
    items_len = len(items)
    for _ in xrange(items_len * ratio * 2):
        item = _movie_list[int(numpy.random.random() * _tot)]
        if item in ret:
            continue
        ret[item] = 0
        n += 1
        if n > items_len * ratio:
            break

    return ret

def __predict(user, item, printF=False):

    rui = 0
    if _bias:
        _bu.setdefault(user, 0)
        _bi.setdefault(item, 0)
        rui += _avr + _bu[user] + _bi[item]
    if _svd:
        release_date = release[item]
        f2 = genres[item]
        aux.setdefault(f2, 1)
        timestamp = _times[item]

        _pu.setdefault(user, numpy.zeros((_k, 1)))
        _qi.setdefault(item, numpy.zeros((_k, 1)))
        _pu1.setdefault(user, numpy.zeros((_k1, 1)))
        _qi1.setdefault(release_date, numpy.zeros((_k1, 1)))
        _pu2.setdefault(user, numpy.zeros((_k2, 1)))
        _qi2.setdefault(f2, numpy.zeros((_k1, 1)))
        _qi3.setdefault(f2, numpy.zeros((_k1, 1)))

        # days=compute_days(timestamp, release_date)

        # _pi.setdefault(item, numpy.zeros((_k1, 1)))
        # _qt.setdefault(days, numpy.zeros((_k1, 1)))

        s1 = 0
        s2 = 0
        s3 = 0
        s4 = 0
        if _weight1 != 0:
            s1 = numpy.sum(_pu[user] * _qi[item]) 
        if _weight2 != 0:
            s2 = numpy.sum(_pu1[user] * _qi1[release_date]) 
        if _weight3 != 0:
            s3 = numpy.sum(_pu2[user] * _qi2[f2]) / numpy.sqrt(aux[f2])

        # s4=_bd[f2]
        # s4=numpy.sum(_pi[item]  * _qt[days])

    if printF:
        print(user, s1, s2, s3)
    
    return s1 + s2 + s3 + rui + s4


def recommend_explicit(user):

    rank = {}
    ru = _user_items[user]
    for item in _movie_set:
        if item in ru:
            continue
        rank[item] = __predict(user, item)
    

    return rank.iteritems()


def recommend_implicit(user, n):

    rank = {}
    ru = _user_items[user]
    for item in _movie_set:
        if item in ru:
            continue
        rank[item] = __predict(user, item)
    return heapq.nlargest(n, rank.iteritems(), key=operator.itemgetter(1))

def compute_days(timestamp, release_date):
    days = (timestamp - time.mktime(datetime.datetime.strptime(str(release_date), "%Y%m%d").timetuple())) / 86400
    if days < 0: 
        days = 0
    return days
