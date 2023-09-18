# coding: utf-8



def list_flatten(t):
    ##make a flat list from a list of lists
    return [item for sublist in t for item in sublist]
