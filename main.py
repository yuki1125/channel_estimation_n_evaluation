import csv

import demodulation_method as md
import channel_estimation as chest
from data_creation import MakeReceivedImg

_loop = 1
_gaussSigma = 0.336
_kernelSize = 9
_maxLum = 1
_num_led = 16
boxNoise = [0.0, 0.2, 0.3, 0.4]

print('Processing is about to begin')

for i, _boxNoise in enumerate(boxNoise): 
    # 画像の作成
    mri = MakeReceivedImg(numberOfLED=_num_led, gaussSigma=_gaussSigma, boxNoise=_boxNoise, \
                        kernelSize=_kernelSize, maxLum=_maxLum, offset=False)
    # loop枚数分画像の作成
    # - answer_signals: 正解の点滅パターン
    # - pixel_values: 受信画素値（ノイズ＋ガウスフィルタ)
    answer_signals, pixel_values = mri.create_dataset(loop=_loop)

    # - ch_pro: 擬似逆行列を用いたチャネル推定
    ch_pro = chest.estimate_channel(answer_signals, pixel_values, offset=False)
    # - ch_conv: 単独点滅画像を用いたチャネル推定
    ch_conv = chest.estimate_channel_conv(gaussSigma=_gaussSigma, kernelSize=_kernelSize)

    # - replicas_with_pro: ch_proを用いたレプリカ画像作成
    replicas_with_pro = md.create_replica(numleds=_num_led, ch_pro)
    # - replicas_with_conv: ch_convを用いたレプリカ画像作成
    replicas_with_conv = md.create_replica(ch_conv)

    print('now processing....:', i, '/', len(boxNoise))
    zf_ber_pro = md.modulation_zf(pixel_values, answer_signals, ch_pro, loop=_loop, offset=False)
    zf_ber_conv = md.modulation_zf(pixel_values, answer_signals, ch_conv, loop=_loop)
    mld_ber_pro = md.modulation_mld(pixel_values, answer_signals, ch_pro, replicas_with_pro, loop=_loop)
    #mld_ber_conv = md.modulation_mld(pixel_values, answer_signals, ch_conv, replicas_with_conv, loop=_loop)
    ber_zf.append(zf_ber_pro)
    ber_mld.append(mld_ber_pro)

print('--'*15)
print('Iteration:', _loop)
print('Gauss_variance:', _gaussSigma)
#print('Noise:', _boxNoise)
print('--'*15)
print('BER-ZF-pro:', zf_ber_pro)
print('BER-ZF-conv:', zf_ber_conv)
print('BER-MLD-pro:', mld_ber_pro)
