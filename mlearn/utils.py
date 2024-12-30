#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 22:30:18 2024

@author: romaric
"""

import numpy as np


def mean_squared_error(y_true, y_pred, squared=True):
    """Calcule l'erreur quadratique moyenne"""
    if y_true.shape != y_pred.shape:
        raise ValueError("`y_true` and `y_pred` must have same shape")
    mse = np.mean((y_true - y_pred) ** 2)
    return (mse if squared else np.sqrt(mse))
    