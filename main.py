import csv

import modulation_method as md
import channel_estimation as chest
from data_creation import MakeReceivedImg

_loop = 100
_gaussSigma = 0.336
_kernelSize = 9
_maxLum = 1
boxNoise = [0.0, 0.2, 0.3, 0.4]


f_mld = open('ber_mld.csv', 'w')
f_zf = open('ber_zf.csv', 'w')
writer_mld = csv.writer(f_mld, lineterminator='\n')
writer_zf = csv.writer(f_zf, lineterminator='\n')
ber_mld = []
ber_zf = []
print('Processing is about to begin')

for i, _boxNoise in enumerate(boxNoise): 
    mri = MakeReceivedImg(gaussSigma=_gaussSigma, boxNoise=_boxNoise, \
                        kernelSize=_kernelSize, maxLum=_maxLum, offset=False)
    answer_signals, pixel_values = mri.create_dataset(loop=_loop)

    ch_pro = chest.estimate_channel(answer_signals, pixel_values, offset=False)
    ch_conv = chest.estimate_channel_conv(gaussSigma=_gaussSigma, kernelSize=_kernelSize)

    replicas_with_pro = md.create_replica(ch_pro)
    replicas_with_conv = md.create_replica(ch_conv)

    print('now processing....:', i, '/', len(boxNoise))
    zf_ber_pro = md.modulation_zf(pixel_values, answer_signals, ch_pro, loop=_loop, offset=False)
    zf_ber_conv = md.modulation_zf(pixel_values, answer_signals, ch_conv, loop=_loop)
    mld_ber_pro = md.modulation_mld(pixel_values, answer_signals, ch_pro, replicas_with_pro, loop=_loop)
    #mld_ber_conv = md.modulation_mld(pixel_values, answer_signals, ch_conv, replicas_with_conv, loop=_loop)
    ber_zf.append(zf_ber_pro)
    ber_mld.append(mld_ber_pro)

writer_mld.writerow(ber_mld)
writer_zf.writerow(ber_zf)
f_mld.close()
f_zf.close()
print('--'*15)
print('Iteration:', _loop)
print('Gauss_variance:', _gaussSigma)
#print('Noise:', _boxNoise)
print('--'*15)
print('BER-ZF-pro:', zf_ber_pro)
print('BER-ZF-conv:', zf_ber_conv)
print('BER-MLD-pro:', mld_ber_pro)
#print('BER-MLD-conv:', mld_ber_conv)