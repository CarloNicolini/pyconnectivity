# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 12:49:22 2014

@author: carlo
"""
from copy import deepcopy


class DisjointSet(object):

    def __init__(self, size=None):
        if size is None:
            self.membership = {}  # maps a member to the group's membership
            # maps a group membership to the group (which is a set)
            self.group = {}
            self.oldgroup = {}
            self.oldmembership = {}
        else:
            self.group = {i: set([i]) for i in range(0, size)}
            self.membership = {i: i for i in range(0, size)}
            self.oldgroup = {i: set([i]) for i in range(0, size)}
            self.oldmembership = {i: i for i in range(0, size)}

    def add(self, a, b):
        self.oldgroup = deepcopy(self.group)
        self.oldmembership = deepcopy(self.membership)

        membershipa = self.membership.get(a)
        membershipb = self.membership.get(b)
        if membershipa is not None:
            if membershipb is not None:
                if membershipa == membershipb:
                    return  # nothing to do
                groupa = self.group[membershipa]
                groupb = self.group[membershipb]
                if len(groupa) < len(groupb):
                    a, membershipa, groupa, b, membershipb, groupb = b, membershipb, groupb, a, membershipa, groupa
                groupa |= groupb
                del self.group[membershipb]
                # qui sembra mancare un pezzo di codice
                for k in groupb:
                    self.membership[k] = membershipa
            else:
                self.group[membershipa].add(b)
                self.membership[b] = membershipa
        else:
            if membershipb is not None:
                self.group[membershipb].add(a)
                self.membership[a] = membershipb
            else:
                self.membership[a] = self.membership[b] = a
                self.group[a] = set([a, b])

    def connected(self, a, b):
        membershipa = self.membership.get(a)
        membershipb = self.membership.get(b)
        if membershipa is not None:
            if membershipb is not None:
                return membershipa == membershipb
            else:
                return False
        else:
            return False

    def undo(self):
        self.group = self.oldgroup.copy()
        self.membership = self.oldmembership.copy()


def renumber(dictionary):
    """
    Renumber the values of the dictionary from 0 to n
    """
    count = 0
    ret = dictionary.copy()
    new_values = dict([])

    for key in dictionary.keys():
        value = dictionary[key]
        new_value = new_values.get(value, -1)
        if new_value == -1:
            new_values[value] = count
            new_value = count
            count += 1
        ret[key] = new_value

    return ret


def renumber_sort(dictionary):
    """
    This function has the membership as input and output the membership
    where the communities number are ordered by the number of nodes in that community
    """
    ds = {}
    for u, v in dictionary.iteritems():
        if v not in ds.keys():
            ds[v] = []
        ds[v].append(u)

    S = dict(
        zip(range(0, len(ds)), sorted(ds.values(), key=len, reverse=True)))
    M = {}

    for u, vl in S.iteritems():
        for v in vl:
            M[v] = u
    return M


def test1():
    import networkx as nx
    G = nx.Graph(nx.karate_club_graph())
    par = nx.Graph()
    par.add_edges_from(G.edges()[0:4])

    ds = DisjointSet()
    for e in par.edges():
        ds.add(e[0], e[1])

    print ds.group
    mi, pi = 0, 0
    for x in ds.group.values():
        ni = len(x)
        pi += ni * (ni - 1) / 2
        mi += G.subgraph(x).number_of_edges()

    print "mi=", mi, "pi=", pi
    print ds.group


def test2():
    x = DisjointSet()
    x.add(0, 1)
    x.add(0, 2)
    x.add(2, 3)
    x.add(3, 4)
    x.add(3, 50)
    x.undo()
    print x.membership
    print x.group

if __name__ == "__main__":
    ds = DisjointSet()
    ds.add(0, 1)
    ds.add(0, 2)
    ds.add(0, 3)
    ds.add(4, 5)
    ds.add(5, 6)
    ds.add(5, 10)
    print ds.membership, "\n", ds.group
    ds.add(3, 10)
    print ds.membership, "\n", ds.group
