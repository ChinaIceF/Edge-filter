import os
import sys
import matplotlib.image as mpimg
import math

def get_size_text(size):
  level_text = ["B", "K", "M", "G", "T", "P", "E", "Z"]
  level = int(math.log(size, 1024))
  size_divided = round(size / (1024 ** level), 1)
  return str(size_divided)+level_text[level]

def get_img_info(filename):
  '''
  if len(sys.argv) > 1 :
    filename = sys.argv[1]
  else :
    filename = "test.bmp"
  '''

  
  #print("\tLoading image..")

  #  读取文件
  image_depth_text = ["0","Bin","Unknown","RGB","RGBA"]
  image_loaded = mpimg.imread(filename)
  image_y, image_x, image_depth = image_loaded.shape
  image_depth = image_depth_text[image_depth]
  #print(image_loaded.shape)

  #  获得文件大小
  file_size = get_size_text(os.path.getsize(filename))
  #print(file_size)

  #  获得文件类型
  file_type = "." + filename.split(".")[-1]
  #print(file_type)

  #  压缩文件地址长度
  file_path_limit = 25
  if len(filename) > file_path_limit:
    zipped_path = filename[:int(file_path_limit)-2] + "..." + filename[-(int(file_path_limit)-1):]
  else:
    zipped_path = filename

  print("")

  print("\tFull Path")
  print("\t  "+zipped_path)
  print("")
  print("\tDigital Infomation")
  print("\t  Title   |\tSize\tType\tX\tY\tChannel")
  print("\t  Value   |\t"+file_size+"\t"+file_type+"\t"+str(image_x)+"\t"+str(image_y)+"\t"+str(image_depth))
