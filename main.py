import argparse
# import numpy as np
import time

from scipy.io import loadmat

from visualization import *

parser = argparse.ArgumentParser(description='Let"s make visualizations')

parser.add_argument('-data', metavar='directory_path', default="./data" , type=str,
                    help='path where all mat files are located')
parser.add_argument('-filter', default="*.mat" , type=str,
                    help='filter how to load pathes')
parser.add_argument('-mode', default= "watch", type = str, help = " (watch)/(generate) video?")
parser.add_argument('-output', default=None, type=str,
                    help='filename how to save gif of movie')

parser.add_argument("-type", default= "gif", type =str, help = "choose gif/avi")

args = parser.parse_args()

if __name__ == "__main__":
    # print(vars(args)["data"])
    print(args.data)
    print(args.filter)
    print(args.output)

    frames = []
    for filename in glob.glob(os.path.join(args.data, args.filter))[0:5]: # hope here all files are sorted
        print(filename)

        start_time = time.clock()
        frame = loadmat(filename)
        frame = frame["TwoD_spectr_UP2"] #what key?!
        frames.append(frame)
        print("time opening:", time.clock() - start_time)

        start_time = time.clock()
        __ = indexing(frame)
        print("time creating pcd:", time.clock() - start_time)
    frames = np.array(frames)

    animateFramesNative(frames, directory_output= args.output)

    if args.output:
        if args.type == "gif":
            make_gif(directory = args.output)
        elif args.type == "avi":
            make_video(directory= args.output)
        else:
            print("sorry, can't find this type of output, use gif or avi")












    # print(globals())
