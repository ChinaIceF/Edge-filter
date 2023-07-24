from image_resizer import *
import numpy
import numba
from numba import *
from PIL import Image

#  使用 CPU 进行卷积 ————————————————————————————————
@jit(nopython = True, parallel = True)
def CNN_numba(image ,kernel,result_array):
  image_y, image_x, channels = image.shape
  if channels == 4:
    channels = 3
    image = image
    result_array = result_array[:,:,0:3]

  limit_y, limit_x = np.array(image.shape)[0:2] - np.array(kernel.shape[0:2])  #  下标上限
  
  kernal_with_channels = kernel
  result = result_array

  for y in prange(0,limit_y,1):
    #if y % 100 == 0:
    #  print("\r",round(y/limit_y*100,3),"%")
    for x in prange(0,limit_x,1):
      tmp_image_part = image[y:y+kernel.shape[0],x:x+kernel.shape[1],0:3]
      tmp_point_result = tmp_image_part * kernal_with_channels
      result[y][x][0:3] = np.mean(tmp_point_result)#, axis = (0, 1))

  return result

#  使用 CPU 并行计算初步降噪 ————————————————————————————
@jit(nopython = True, parallel = True)
def denoise(image,result_array):
  image_y, image_x = image.shape
  
  #  image 应该比原来的图片宽一圈，防止溢出，周围的点全是白的
  #  image 是一个 binarized 图片，只有 0 与 1  
  #  降噪原理：如果一个像素相邻的四个像素全是黑色，则这是一个独立点，把它清除，反之输出到结果
  output = result_array
  noise_amount = 0
  
  for y in prange(1,image_y - 1,1):
    for x in prange(1,image_x - 1,1):
      checksum = image[y+1][x] + image[y-1][x] + image[y][x+1] + image[y][x-1]
      if checksum > 0:
        output[y-1][x-1] = image[y][x]
      else:
        output[y-1][x-1] = 0
        if image[y][x] == 1 :
          noise_amount += 1

  return output, noise_amount


#  二级降噪 —————————————————————————————————————
@jit(nopython = True)#, parallel = True)
def denoise_level2(image,original_image,result_array,max_noise_size):
  image_y, image_x = image.shape
  noise_amount = 0
  
  #  image 应该比原来的图片宽一圈，防止溢出，周围的值为-99
  #  传入的 image 是 binarized 图片，只有 0 和 1，且背景是黑色（0），1代表有意义
  for y in prange(1,image_y - 1,1):
    #print(y, image_y)
    for x in prange(1,image_x - 1,1):
      if image[y,x] == 1:  #  如果这个点是白的，那么去它的周围找
       
        this_noise_size = 1
        searched_locate = [[-1,-1]]
        locate_to_search = [[-2,-2]]
        next_turn_locate_to_serach = [[-1,-1]]
        next_turn_locate_to_serach.append([y,x])
        
        while not locate_to_search == [[-1,-1]]:
          #print(locate_to_search)
          locate_to_search = next_turn_locate_to_serach
          next_turn_locate_to_serach = [[-1,-1]]
          for L in locate_to_search[1:]:
            for P in ([L[0]-1,L[1]], [L[0]+1,L[1]], [L[0],L[1]-1], [L[0],L[1]+1]):
              if not P in searched_locate:    #  如果这个点没有被搜索到过才能使用
                searched_locate.append(P)  #  记录已经搜索过的点
                if image[P[0],P[1]] == 1 :  #  如果被检测的这个点是白的，那么记录它
                  this_noise_size += 1
                  image[P[0],P[1]] == -1
                  next_turn_locate_to_serach.append(P)
        
        #  如果低于或等于认为是噪声的最大值，删掉它
        #  反之，则输出这些点
        if this_noise_size > max_noise_size:
          noise_amount += this_noise_size
          for P in searched_locate[1:]:
            #result_array[P[0]-1][P[1]-1] = 1
            image[P[0]][P[1]] = 0
            #image[P[0]][P[1]] = 0
        
  return original_image - image, noise_amount
  
#  主进程开始 ————————————————————————————————————
if __name__ == "__main__":

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

  print(" Processing...")
  #  识别图像边缘
  image_protected = np.ones(np.array(image_loaded.shape) + np.array([2,2,0]))
  image_protected[1:-1, 1:-1] = image_loaded
  laplas_kernel = Kernel([[0,-0.25,0],[-0.25,1,-0.25],[0,-0.25,0]])
  image_edge_recongnized = CNN_numba(image_protected, 
                                   np.stack((laplas_kernel.arg,) * 3, axis = -1), 
                                   np.ones(image_protected.shape, dtype = "float32"))
  image_edge_recongnized = image_edge_recongnized[1:-1,1:-1]
  
  
  #  循环  让用户得以调整参数
  while True:
    test = np.array(image_edge_recongnized)
    bias, threshold, max_noise_size = input("(bias, threshold(0-255)), max_noise_size(int) >>> ").split(" ")
    bias = float(bias)
    threshold = float(threshold)
    max_noise_size = int(max_noise_size)
    
    time_start = time.time()
    
    pil_image_calculated = Image.fromarray(np.uint8(test * 255 + bias))
    pil_image_calculated.save("output_denoised_level_0.png")
    
    image_reloaded = mpimg.imread("output_denoised_level_0.png")
    image_reloaded = image_reloaded.mean(axis = 2)
    image_reloaded = np.array(image_reloaded > threshold/255 , dtype = "int8")
    
    #  第一级降噪，使用 CPU ，去除单点的噪点
    print("\n\tTrying to denoise with CPU...(Level 1)")
    image_ready_to_denoise = np.ones(np.array(image_reloaded.shape) + 2)
    image_ready_to_denoise[1:-1,1:-1] = image_reloaded
    image_denoised, noise_amount = denoise(image_ready_to_denoise, np.ones(np.array(image_reloaded.shape)))
    print("\t  - Removed",noise_amount,"point(s).")
    print("\t    Weight:",round(100 * noise_amount / (image_y*image_x), 3),"%")
    
    pil_image_calculated = Image.fromarray(np.uint8((image_denoised)* 255))
    pil_image_calculated.save("output_denoised_level_1.png")

    
    #  第二级降噪
    print("\n\tTrying to denoise with CPU...(Level 2)")
    image_denoised = image_denoised[1:-1,1:-1]
    image_ready_to_denoise = np.zeros(np.array(image_denoised.shape) + 2)
    image_ready_to_denoise[1:-1,1:-1] = image_denoised
    image_denoised_level2, noise_amount = denoise_level2(image_ready_to_denoise, numpy.array(image_ready_to_denoise), np.zeros(np.array(image_denoised.shape)), max_noise_size)
    print("\t  - Removed",noise_amount,"point(s).")
    print("\t    Weight:",round(100 * noise_amount / (image_y*image_x), 3),"%")
    
    
    pil_image_calculated = Image.fromarray(np.uint8((image_denoised_level2[1:-1,1:-1])* 255))
    pil_image_calculated.save("output_denoised_level_2.png")

    print("\n\tTime costs:", time.time() - time_start,"sec.")





