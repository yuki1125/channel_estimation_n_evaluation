import numpy as np

from data_creation import MakeReceivedImg

def estimate_channel(led_pattern, pixel_value, offset=False):
    """
    提案手法でのチャネル推定
    """
    
    #led, pix = mri.create_dataset(loop=10)
    inv_led = np.linalg.pinv(led_pattern)
    len_loop, len_led = led_pattern.shape
    
    if offset:
        total_led = len_led - 1
    else:
        total_led = len_led

    estimated_channel = np.array([[0 for i in range(len_led)] for j in range(total_led)], dtype='float32')
    
    for idx_led in range(total_led):
        tmp_led = []
       
        for idx_loop in range(len_loop):
            tmp_led.append(pixel_value[idx_loop, idx_led])
            
        # tmp_channel(len_led, 1)の配列
        tmp_channel = np.dot(inv_led, tmp_led)
        
        for i in range(len_led):
            estimated_channel[idx_led, i] = tmp_channel[i]

    return estimated_channel

def estimate_channel_conv(numleds=16, numimages=50, gaussSigma=0.4, boxNoise=0.2, kernelSize=9, maxLum=1, offset=False):
    """
    個別点灯画像のnumimages枚の平均を用いたチャネル推定
    """
    from data_creation import MakeReceivedImg
    mri_est = MakeReceivedImg(numberOfLEDs=numleds, boxNoise=boxNoise, gaussSigma=gaussSigma, kernelSize=kernelSize)
    estimated_channel = np.array([[0 for i in range(numleds)] for j in range(numleds)], dtype='float32')
    
    for led in range(numleds):
        leds = np.zeros(numleds, dtype='int32')
        leds[led] = 1
        total_pix_val = [0]
        
        for _ in range(numimages):
            pix = mri_est.Filtering(leds=leds)
            noise = mri_est.GetNoise()
            
            if offset: # Offsetで全体画素値を底上げ
                pix_val = pix + noise + 10*0.01
            else:
                pix_val = pix + noise
                
            total_pix_val += pix_val / numimages
        
        for i in range(numleds):
            estimated_channel[led, i] = total_pix_val[i]
    
    return estimated_channel