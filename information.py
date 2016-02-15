#    This file is part of pyConnectivity
#
#    pyConnectivity is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 2 of the License, or
#    (at your option) any later version.
#
#    pyConnectivity is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with pyConnectivity. If not, see <http://www.gnu.org/licenses/>.
#
#    Copyright 2014 Carlo Nicolini <carlo.nicolini@iit.it>
#
#    Further informations and API are available at
#    http://perso.crans.org/aynaud/communities/api.html

"""
This module implements variation of information.
"""

__author__ = "Carlo Nicolini <carlo.nicolini@iit.it>"
__all__ = ["confusion_matrix", "normalized_mutual_information",
           "variation_of_information", "adjusted_rand_index"]

from pyconnectivity import nx, np, copy, logging, statistic, community
# http://thirdorderscientist.org/homoclinic-orbit/2013/9/5/how-to-compare-communities-using-variation-of-information
import math
import string
import sys
import fileinput
import numpy as np
import copy

# http://thirdorderscientist.org/homoclinic-orbit/2013/9/5/how-to-compare-communities-using-variation-of-information


def confusion_matrix(partA, partB):
    if len(partA) != len(partB):
        raise Exception("Incompatible partitions!")
    N = len(partA)
    Ca = len(np.unique(partA.values()))
    Cb = len(np.unique(partB.values()))
    # important to bring all the symbol to the same meaning, renumbering is
    # important
    partitionA = community.__renumber(copy.copy(partA))
    partitionB = community.__renumber(copy.copy(partB))

    Ca, Cb = {}, {}
    for c in partitionA.values():
        Ca[c] = []
    for c in partitionB.values():
        Cb[c] = []

    for n, c in partitionA.iteritems():
        Ca[c].append(n)

    for n, c in partitionB.iteritems():
        Cb[c].append(n)

    C = np.zeros([len(Ca), len(Cb)])

    for i in range(0, len(Ca.keys())):
        for j in range(0, len(Cb.keys())):
            C[i][j] = len(set(Ca[i]).intersection(set(Cb[j])))
    return C


def informations(C, N):
    """
    Compute the normalized mutual information
    """
    # Compute Ni,Nj
    Ni = C.sum(1)
    Nj = C.sum(0)
    # Compute numerator of mutual information
    num = 0.0
    for i in range(0, C.shape[0]):
        for j in range(0, C.shape[1]):
            if C[i][j] == 0.0:
                num += 0.0
            else:
                num += C[i][j] * np.log((C[i][j] * float(N)) / (Ni[i] * Nj[j]))

    # Compute denominator of mutual information
    Ha, Hb = 0.0, 0.0
    d1 = (Ni * (np.log(Ni) - np.log(N))).sum()
    d2 = (Nj * (np.log(Nj) - np.log(N))).sum()
    # Normalized mutual information
    nmi = -2.0 * num / (d1 + d2)
    # Compute entropy of first and second set
    Ha = -(Ni / N * (np.log(Ni) - np.log(N))).sum()
    Hb = -(Nj / N * (np.log(Nj) - np.log(N))).sum()
    return nmi, Ha, Hb


def variation_of_information(partitionA, partitionB):
    C = confusion_matrix(partitionA, partitionB)
    nmi, Ha, Hb = informations(C, len(partitionA))
    # In case both partitions are composed of only one community
    if C.shape == (1, 1):
        vi = 0.0
        nmi = 1.0
        return vi
    # Compute variation of information as H(X)+H(y)-2I(X|Y)
    vi = (1.0 - nmi) * (Ha + Hb)
    return vi


def normalized_mutual_information(partitionA, partitionB):
    C = confusion_matrix(partitionA, partitionB)
    nmi, Ha, Hb = informations(C, len(partitionA))
    return nmi


def normalized_variation_of_information(partitionA, partitionB):
    vi = variation_of_information(partitionA, partitionB)
    return vi / np.log2(len(partitionA))


def adjusted_rand_index(partitionA, partitionB):
    C = confusion_matrix(partitionA, partitionB)
    numRows, numCols = C.shape
    rowChoiceSum = 0
    columnChoiceSum = 0
    totalChoiceSum = 0
    total = 0
    def pairs(x):
        return x * (x - 1 ) / 2

    for i in range(0,numRows):
        rowSum = 0
        for j in range(0, numCols):
            rowSum += C[i, j]
            totalChoiceSum += pairs(C[i, j])
        total += rowSum
        rowChoiceSum += pairs(rowSum)

    for j in range(0,numCols):
        columnSum = 0
        for i in range(0, numRows):
            columnSum += C[i, j]
        columnChoiceSum += pairs(columnSum)

    rowColumnChoiceSumDivTotal = rowChoiceSum * columnChoiceSum / pairs(total);
    return (totalChoiceSum - rowColumnChoiceSumDivTotal) / ((rowChoiceSum + columnChoiceSum) / 2 - rowColumnChoiceSumDivTotal)


class PartitionInformation:
    def __init__(self, partA, partB):
        if len(partA) != len(partB):
            raise Exception("Incompatible partitions!")
        N = len(partA)
        Ca = len(np.unique(partA.values()))
        Cb = len(np.unique(partB.values()))
        # important to bring all the symbol to the same meaning, renumbering is
        # important
        partitionA = self.renumber(partA)
        partitionB = self.renumber(partB)

        Ca, Cb = {}, {}
        for c in partitionA.values():
            Ca[c] = []
        for c in partitionB.values():
            Cb[c] = []

        for n, c in partitionA.iteritems():
            Ca[c].append(n)

        for n, c in partitionB.iteritems():
            Cb[c].append(n)

        C = np.zeros([len(Ca), len(Cb)])

        for i in range(0, len(Ca.keys())):
            for j in range(0, len(Cb.keys())):
                C[i][j] = len(set(Ca[i]).intersection(set(Cb[j])))
        self.confusion_matrix = copy.copy(C)

        # Compute Ni,Nj
        Ni = C.sum(1)
        Nj = C.sum(0)
        # Compute numerator of mutual information
        num = 0.0
        for i in range(0, C.shape[0]):
            for j in range(0, C.shape[1]):
                if C[i][j] == 0.0:
                    num += 0.0
                else:
                    num += C[i][j] * \
                        np.log((C[i][j] * float(N)) / (Ni[i] * Nj[j]))

        # Compute denominator of mutual information
        Ha, Hb = 0.0, 0.0
        d1 = (Ni * (np.log(Ni) - np.log(N))).sum()
        d2 = (Nj * (np.log(Nj) - np.log(N))).sum()
        # Normalized mutual information
        self.nmi = -2.0 * num / (d1 + d2)
        # Compute entropy of first and second set
        self.Ha = -(Ni / N * (np.log(Ni) - np.log(N))).sum()
        self.Hb = -(Nj / N * (np.log(Nj) - np.log(N))).sum()

        if C.shape == (1, 1):
            self.vi = 0.0
            self.nmi = 1.0
        else:
            # Compute variation of information as H(X)+H(y)-2I(X|Y)
            self.vi = (1.0 - self.nmi) * (self.Ha + self.Hb)

        self.nvi = self.vi / np.log2(len(partitionA))

    def renumber(self, partition):
        count = 0
        ret = partition.copy()
        new_values = dict([])

        for key in partition.keys():
            value = partition[key]
            new_value = new_values.get(value, -1)
            if new_value == -1:
                new_values[value] = count
                new_value = count
                count += 1
            ret[key] = new_value
        return ret
