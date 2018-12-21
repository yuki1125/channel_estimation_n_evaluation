import math
from itertools import product

import numpy as np
from tqdm import tqdm

def modulation_zf(pixelValue, answers, gauss, loop, offset=False):
    """
    推定したチャネルからZFにより復調を行う
    """
    _, len_led = pixelValue.shape
    
    if offset:
        offset_value = gauss[:, len_led]
    
    answers = answers[:, :len_led]
    gauss = gauss[:, :len_led]
    inv_gauss = np.linalg.pinv(gauss)

    error = 0
    
    for loop in range(loop):
        
        pixel_value = pixelValue[loop][:]
        
        if offset:
            pixel_value = pixel_value - offset_value
        
        answer = answers[loop][:]
        estimated_signals = np.dot(inv_gauss, pixel_value)
        estimated_signals_copy = estimated_signals.copy()
        estimated_signals_copy = np.where(estimated_signals_copy > 0.5, 1, 0)
        error += np.sum(estimated_signals_copy != answer)
    
    return error / (loop*len_led)


def create_replica(gauss, offset=False):
    """
    推定したチャネルからレプリカ画像を作成する
    """
    
    _, len_led = gauss.shape
    
    if offset:
        offset_value = gauss[:, len_led]
    
    gauss = gauss[:, :len_led]
    
    # 全点滅パターンの格納
    if offset:
        pixel_values = np.array([np.dot(gauss, i) - offset_value \
                                 for i in product(range(2), repeat=len_led)])
    else:
        pixel_values = np.array([np.dot(gauss, i) for i in product(range(2), repeat=len_led)])
    
    return pixel_values
    
def modulation_mld(pixelValue, answers, gauss, replicas, loop, offset=False):
    """
    推定したチャネルからMLDにより復調を行う
    """
    
    _, len_led = pixelValue.shape
    num_of_replicas = np.power(2, len_led)
    if offset:
        offset_value = gauss[:, len_led]
    
    answer = np.array(answers[:, :len_led], dtype='int32')
    gauss = gauss[:, :len_led]
    
    error = 0

    for idx in tqdm(range(loop)):
        diff = [(replicas[i] - pixelValue[idx]) for i in range(num_of_replicas)]
        diff_euclid = np.power(diff, 2)
        diff_sum = np.sqrt(np.sum(diff_euclid, axis=1))
        diff_min = format(np.argmin(diff_sum), 'b').zfill(len_led)
        estimated_signals = np.array([int(i) for i in diff_min])
        error += np.sum(estimated_signals != answer[idx])

    return error/(len_led*loop)
    
    
    