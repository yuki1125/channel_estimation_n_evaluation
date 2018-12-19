import numpy as np
import math

class MakeReceivedImg(object):
    def __init__(self, numberOfLEDs=16, gaussSigma=0.45, kernelSize=9, maxLum=1, minLum=0, boxNoise=0.3, offset=False):
        """
        Initial functions of this class
        """
        self.numberOfLEDs = numberOfLEDs
        self.sqrtnumberofled = math.sqrt(numberOfLEDs)
        self.gaussSigma = gaussSigma
        self.kernelSize = kernelSize
        self.maxLum = maxLum
        self.minLum = minLum
        self.boxNoise = boxNoise
        self.offset = offset
        if offset:
            self.add_led = 1
        else:
            self.add_led = 0
    
    def RandomLEDs(self):
        """
        Making randomly blinking LED and store them in array(numpy) 
        numberOfLEDs: Number of LEDs. Default = 16
        max_lum_value: maximum luminance value of LEDs
        min_lum_value: minimum luminance value of LEDs
        """
        # Generating random integers [0,1]
        blinking_leds = np.random.randint(0, 2, self.numberOfLEDs)
        
        # aplly max/min luminance value depending on its blinking condition
        for index, value in enumerate(blinking_leds):
          if value == 0:
            blinking_leds[index] = self.minLum
          else:
            blinking_leds[index] = self.maxLum

        # [sqrt(len(blinking_leds)]
        led_len = math.sqrt(len(blinking_leds))
        assert led_len % 2 == 0, 'Expected: 0 == led_len%2 \n Actual: 1 == led_len%2'

        # change dtype float to int
        led_len = int(led_len)

        # reshape led_len to two-d array from one-d array
        # led_condition = np.reshape(blinking_leds, (led_len, led_len))

        return np.array(blinking_leds)


    def GaussChannelAndInv(self):
        """
        Creating channel parameters based on Gauss function
        """

        y_scale = int(self.sqrtnumberofled)
        x_scale = int(self.sqrtnumberofled)        

        # 配列の確保
        gauss_channel = [[0 for i in range(y_scale * x_scale)] for j in range(y_scale * x_scale)]

        # 自身と他の画素から影響を受ける画素の決定
        for y in range(0, y_scale):
            for x in range(0, x_scale):
                target_pixel = y * x_scale + x  

                # target_pixelへ影響を与えるinfluence_pixelを決定
                for i in range(y_scale):
                    for j in range(x_scale):

                        influence_pixel = i * x_scale + j

                        # ガウシアンフィルの作成
                        distance = (j - x) * (j - x) + (i - y) * (i - y)
                        weight = math.exp(-distance / (2.0 * self.gaussSigma * self.gaussSigma)) \
                                 * (1 / (2.0 * math.pi * self.gaussSigma * self.gaussSigma))

                        if ((j - x) * (j - x)) >= self.kernelSize or ((i - y) * (i - y)) >= self.kernelSize:
                            weight = 0
                        else:
                            gauss_channel[target_pixel][influence_pixel] = weight

                gauss_channel = np.array(gauss_channel)

        # 逆行列の作成
        inv_channel = gauss_channel.copy()
        inv_gauss_channel = np.linalg.inv(inv_channel)
        return gauss_channel, inv_gauss_channel
      
      
    def Filtering(self, leds):
        """
        Aplly filtering to LED arrays created in RandomLEDs
        """
        channel, _ = self.GaussChannelAndInv()
        pixel_values = np.dot(channel, leds)
        
        return pixel_values    
    
  
    def GetNoise(self):
        """
        Generate noise by boxmuller method
        """
        from scipy.stats import uniform
        # 独立した一様分布からそれぞれ一様乱数を生成
        np.random.seed()
        N = self.numberOfLEDs
        rv1 = uniform(loc=0.0, scale=1.0)
        rv2 = uniform(loc=0.0, scale=1.0)
        U1 = rv1.rvs(N)
        U2 = rv2.rvs(N)

        # Box-Mullerアルゴリズムで正規分布に従う乱数に変換
        # 2つの一様分布から2つの標準正規分布が得られる
        X1 = np.sqrt(-2*np.log(U1)) * np.cos(2*np.pi*U2) * self.boxNoise
        #X2 = np.sqrt(-2 * np.log(U1)) * np.sin(2 * np.pi * U2) * boxNoise

        return X1
    
    def ReceivedImg(self):
        """
        Create received image
        """
        leds = self.RandomLEDs()
        pix = self.Filtering(leds)
        noise = self.GetNoise()
        
        if self.offset:
           # offset = np.random.randint(10, 11, self.numberOfLEDs)
            receivedimg = pix + noise + 10*self.maxLum*0.01
            leds = np.append(leds, 1)
        else:
            receivedimg = pix + noise
            
        return receivedimg, leds
    
    def create_dataset(self, loop=100):
        """
        Create dataset
        """
        
        # LED condition (1/0)
        led_condition = np.empty((0))
        pixel_value = np.empty((0))
        
        for _ in range(loop):
            receivedimg, led_condi = self.ReceivedImg()
            pixel_value = np.hstack((pixel_value, receivedimg))
            
            led_condition = np.hstack((led_condition, led_condi))
           
        led_condition_2d = np.reshape(led_condition, (loop, self.numberOfLEDs+self.add_led))
        pixel_value_2d = np.reshape(pixel_value, (loop, self.numberOfLEDs))
        
        return led_condition_2d, pixel_value_2d
