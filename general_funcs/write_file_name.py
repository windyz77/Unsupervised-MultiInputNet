import os

data_dir = "/home/jethong/data/full_data/lytroillum"
path_dir = "full_data/lytroillum/"
path_file_name = '/home/jethong/Desktop/1.txt'
k = [30,0,10,20,40,50,60,6,14,22,28,46,54,27,28,29,31,32,33,3,12,21,39,48,57]
M = [0,1,2,9,10,11,18,19,20]
for single_dir in os.listdir(data_dir):
    # for j in range(9):
        for i in range(len(k)):
            n = str(k[i]+10)
        # if not os.path.exists(path_file_name):
        #     with open(path_file_name,"a") as f:
        #         f.write(path_dir + single_dir + "/" + "inputCam_" + n.zfill(3) + ".png")
                # f.write(k)
        # f = open('/home/hongyongjie/data/file_txt/1.txt','w')
            f = open(path_file_name,"a+")
            f.write(path_dir + single_dir + "/" + "input_Cam" + n.zfill(3) + ".png"+" ")
        f.write('\n')

f.close()