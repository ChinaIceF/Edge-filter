import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
from numba import cuda
import multiprocessing
import numpy as np
import time
import sys
from getfileinfo import *
from PIL import Image

class Kernel(object):
  def __init__(self, arg):
    self.arg = np.array(arg)
    self.y, self.x = self.arg.shape

def binarize_image(image, limit):
  #  根据图片和阈值二值化图片，返回一个二维 np 数组
  color_limit = limit
  image_binarized = np.mean(image, axis = 2)
  image_binarized = np.array(image_binarized > color_limit, dtype = "int8")
  return image_binarized

def colorize_image(image_binarized, depth):
  #  将二值化图片转化为可输出成文件的图片，depth 指通道数，一般为 3 或 4
  image_with_color = np.stack((image_binarized,) * depth, axis = -1)
  return image_with_color

def generate_image(image, name):
  #  把图片输出成文件
  pil_image_calculated = Image.fromarray(np.uint8(image*255))
  pil_image_calculated.save(name,"png")
  

def resize_image_directly(image):
  #  直接把图片放大两倍
  image_y, image_x, channels = image.shape
  result = np.ones([image_y * 2, image_x * 2, channels])
  for y in range(image_y):
    #print("\r无处理放大图片中",round(y*100/image_y, 3),"%             ",end="")
    print("\r\tResizing ...",round(y*100/image_y, 3),"%             ",end="")
    for x in range(image_x):
      tmp = image[y][x]
      result[y*2:y*2+2,x*2:x*2+2] = np.array([[tmp,tmp],[tmp,tmp]])
  
  print("\r\tResizing ... Done.             ")
  return result
  
def resize_image(image, times):
  #  先根据放大的倍数 times 得到放大后的 x，y
  original_x, original_y = np.shape(image)
  resized_x = int(original_x * times)
  resized_y = int(original_y * times)
  
  #  给图像添加一圈保护层，即长宽各加4像素，以免计算时溢出
  image_protected = np.ones(np.array(image.shape) + 4)
  image_protected[2:-2, 2:-2] = image

  #  生成在原图的 x，y 采样的坐标序列
  range_x = np.arange(0, original_x, 1/times)
  range_y = np.arange(0, original_y, 1/times)
  
  #  生成放大后的空图片
  shape_resized = [len(range_y), len(range_x)]
  resized_image = np.ones(shape_resized)
  print(resized_image.shape)
  
  #  在原图上取样
  for index_y in range(len(range_y)):
    y = range_y[index_y]
    #print("\r处理图片中",round(index_y*100/len(range_y), 3),"%",end="")
    for index_x in range(len(range_x)):
      x = range_x[index_x]
      
      #  求出相对整像素点的位移
      dx = x - int(x)
      dy = y - int(y)
      
      #  切出图片中需要取样的那一块
      tmp_image = np.array(image[int(y):int(y)+2 ,int(x):int(x)+2])
      
      #  利用矩阵，算出每个区域的权重
      mat_x = np.mat([[1-dx, dx]])
      mat_y = np.mat([[1-dy], [dy]])
      weight = np.array(np.dot(mat_y,mat_x))
      
      #  通过权重计算这一点的平均色
      color = np.sum(tmp_image * weight, axis = (0,1))
      resized_image[index_y][index_x] = color

  return resized_image

def CNN(image ,kernel):
  image_y, image_x, channels = image.shape
  if channels == 4:
    channels = 3
    image = image[:,:,0:3]
  limit_y, limit_x = np.array(image.shape)[0:2] - np.array(kernel.arg.shape)  #  下标上限
  
  kernal_with_channels = np.stack((kernel.arg,) * 3, axis = -1)
  
  result = np.ones(image.shape)  

  for y in range(0,limit_y,1):
    print("\r",y/limit_y,end = "")
    for x in range(0,limit_x,1):
      tmp_image_part = image[y:y+kernel.y,x:x+kernel.x,0:3]
      tmp_point_result = tmp_image_part * kernal_with_channels
      result[y][x][0:3] = tmp_point_result.mean(axis = (0, 1))

  return result

@cuda.jit(device=True)
def GPU_calculate(tmp_min, tmp_max, tmp_avg, origin_color, y, x, z):



  if not tmp_max - tmp_min == 0 :
    tmp_result = tmp_min + (tmp_max - tmp_min) / (1 + 10 ** ((-arg_recolorize_slope) * (tmp_avg - arg_recolorize_confidence)))
  else:
    tmp_result = tmp_avg
  
  return (tmp_result + origin_color) / 2
  '''
  if tmp_result > origin_color:
    return tmp_result
  else:
    return origin_color
  '''
@cuda.jit
def recolorize_GPU(image, result_image):

  height, width, channels = image.shape
  deprotected_image = image[1:-1,1:-1]


  startX = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
  startY = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
  gridX = cuda.gridDim.x * cuda.blockDim.x;
  gridY = cuda.gridDim.y * cuda.blockDim.y;
  
  
  for z in range(channels):
    for x in range(startX + 1, width - 1, gridX):
      for y in range(startY + 1, height - 1, gridY):

        tmp_list = image[y - 1:y + 2, x - 1:x + 2, z]
        tmp_avg = ((tmp_list[0,0] + tmp_list[0,1] + tmp_list[0,2] + 
                        tmp_list[1,0] + tmp_list[1,1]*0 + tmp_list[1,2] + 
                        tmp_list[2,0] + tmp_list[2,1] + tmp_list[2,2] ) / 8 )
        A = (tmp_list[0,0], tmp_list[0,1], tmp_list[0,2],
              tmp_list[1,0],                   tmp_list[1,2],
              tmp_list[2,0], tmp_list[2,1], tmp_list[2,2])

        tmp_min = min(A)
        tmp_max = max(A)
        '''
        for each in A:
          if each > tmp_max:
            tmp_max = each
          if each < tmp_min:
            tmp_min = each
        '''
        

        result_image[y-1][x-1][z] = GPU_calculate(tmp_min, tmp_max, tmp_avg, image[y][x][z], y, x, z)




def recolorize_way1(image):
  #  注意，这里的 image 应是一个已经添加过保护区的放大两倍后的图片
  image_y, image_x, channels = image.shape
  deprotected_image = image[1:-1,1:-1]
  recolorize_alpha = np.ones([image_y - 2, image_x - 2, channels])  #  输出要删除保护区
  recolorize_color = np.ones([image_y - 2, image_x - 2, channels])  #  输出要删除保护区
  
  for y in range(1,image_y - 1,1):
    #print("\r重着色 - 方案一 处理图片中",round(y*100/image_y, 3),"%             ",end="")
    print("\r\tRecolorizing ...",round(y*100/image_y, 3),"%             ",end="")
    for x in range(1,image_x - 1,1):
      for z in range(channels):
        tmp_list = [image[y-1][x-1][z], image[y-1][x+1][z], image[y+1][x-1][z], image[y+1][x+1][z],
                          image[y][x-1][z], image[y][x+1][z], image[y-1][x][z], image[y+1][x][z]]
        tmp_avg = sum(tmp_list) / len(tmp_list)
        
        #  6.19  算法
        #tmp_result = min(tmp_list) + (1 / (1 + 10 ** ((-arg_recolorize_slope) * (tmp_avg - arg_recolorize_confidence)))
        
        #  6.25  算法 2
        if not max(tmp_list) - min(tmp_list) == 0 :
          tmp_result =min(tmp_list) + (max(tmp_list) - min(tmp_list)) / (1 + 10 ** ((-arg_recolorize_slope) * (tmp_avg - arg_recolorize_confidence)))
        else:
          tmp_result = tmp_avg
        
        recolorize_color[y-1][x-1][z] = max([tmp_result, image[y][x][z]])
        
        #  6.25  算法
        '''
        _max = max(tmp_list)
        _min = min(tmp_list)
        _range = _max - _min
        _range_limit = 0
        recolorize_color = tmp_avg
        
        if _range > _range_limit :
          tmp_result = (1/(1 + 10 ** ((-arg_recolorize_slope) * 
                              ((tmp_avg - _min) / (_max - _min) - arg_recolorize_confidence))))
          recolorize_alpha[y-1][x-1] = tmp_result
        else :
          recolorize_alpha[y-1][x-1] = 0
        
        '''

  
  print("\r\tRecolorizing ... Done.             ")
  return recolorize_color  #* deprotected_image


if __name__ == "__main__":

  #  ____  参数  ____
  #  重绘制的自信度，较高的值会产生锐利的边缘和饱和的颜色
  arg_recolorize_confidence = 0.5

  #  重绘制的数据压缩强度，较高的值会产生锐利的边缘
  arg_recolorize_slope = 4

  test_arg_min = 1 / (1 + 10 ** ((-arg_recolorize_slope) * (0 - arg_recolorize_confidence)))
  test_arg_max = 1 / (1 + 10 ** ((-arg_recolorize_slope) * (1 - arg_recolorize_confidence)))

  if test_arg_min > 0.01 or test_arg_max < 0.99 :
    print("\n\t___________________  Warning  ____________________")
    print("\t  There is an excepted value of arguments.")
    print("\t  Which means the arguments make the function \n\tunsuitable for this algorithm.")
    print("\tThe min-max value is","\n\t ",tuple([test_arg_min,test_arg_max]))

  if len(sys.argv) > 1 :
    filename = sys.argv[1]
  else :
    filename = "test.png"

  get_img_info(filename)
  print("\n\t_____________________  Processing _____________________")
  print("")

  #  读取文件
  image_loaded = mpimg.imread(filename)
  image_y, image_x, image_depth = image_loaded.shape
  #print(image_loaded.shape)

  #  二值化图片，以0.5灰度为阈值
  '''
  image_binarized = binarize_image(image_loaded, 0.5)
  print(image_binarized.shape)
  '''
  #image_binarized = np.mean(image_loaded, axis = 2)
  image_binarized = image_loaded
  #  无处理放大两倍
  resized_two_times = resize_image_directly(image_binarized)
  generate_image(resized_two_times, name = "temp\\resized.png")

  #  给图像添加一圈保护层，即长宽各加2像素，以免计算时溢出
  image_protected = np.ones(np.array(resized_two_times.shape) + np.array([2,2,0]))
  image_protected[1:-1, 1:-1] = resized_two_times
  #print(image_protected.shape)

  #image_recolorized = recolorize_way1(image_protected)
  #  调用GPU进行计算

  height, width, channels = image_protected.shape

  np_image = np.array(image_protected, dtype = "float32")
  empty_image = np.ones([height - 2, width - 2, channels],dtype = "float32")
  blockdim = (32, 8)
  griddim = (32,16)

  start = time.time()
  input_image = cuda.to_device(np_image)
  output_image = cuda.to_device(empty_image)

  recolorize_GPU[griddim, blockdim](input_image, output_image) 

  output = output_image.copy_to_host()
  dt = time.time() - start

  generate_image(output, "recolorized.png")

  print("\n\t_______________________  Result _______________________")
  get_img_info("recolorized.png")










