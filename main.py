import argparse

from scipy.io import loadmat

from visualization import *

# import numpy as np

parser = argparse.ArgumentParser(description='Let"s make visualizations')

parser.add_argument('-data', metavar='directory_path', default="./data" , type=str,
                    help='path where all mat files are located. or filapath to one mat file to visualise only one spectr')
parser.add_argument('-filter', default="*.mat" , type=str,
                    help='filter how to load pathes if data is directory')
parser.add_argument('-mode', default= "watch", type = str, help = " (watch)/(generate) video?")
parser.add_argument("-num", default = "all", type = str, help = "how many frames visualise, or 'all'(by default) ")
parser.add_argument('-output', default=None, type=str,
                    help='filename how to save gif of movie')
parser.add_argument('--screen_size', default = [1280,720], help = "here you can define size of scren")
#todo update folder output if there are other files.

parser.add_argument('-loop_mode', default = False, type = str, help = 'watch endlessly or one time')
#todo loop mode

parser.add_argument('-key', default= "TwoD_spectr_UP2", type = str, help = "key by which we need to tackle from mat files, by default  TwoD_spectr_UP2 ")
parser.add_argument('-is_manual', default = False, type = bool, help = 'if is_manual == True visualization will be run '
                                                                       'with advanced functionality. press X for step-by-step '
                                                                       'frame changing. A - load viewpoint, B - save viewpoint(viewpoint.json in root),'
                                                                       'Z - change to random frame. A lot of functionality can be added')
parser.add_argument('-is_viewpoint', default = False, type = bool, help = 'load viewpoint or not')

parser.add_argument("-type", default= "gif", type =str, help = "choose gif/avi")
parser.add_argument("-threshold", default = 0.5, type = float, help = "choose threshold for visualization, default 0.5")


args = parser.parse_args()

if __name__ == "__main__":
    # print(vars(args)["data"])
    print("директория, откуда берем файлы mat: ", args.data)
    print("фильтр, которым обрабатываем список файлов из директории: ", args.filter)
    print("директория, куда сохраняем выходной файл и промежуточные скрины: ", args.output)
    print("порог визаулизации ", args.threshold )
    print(f"визулизирую {args.num} кадров/ы")
    print(f"из мат файлов достаю вот такие ключи {args.key}")
    print(f"размер окна  визуализации: {args.screen_size}")
    print(f'run in advnced mode {args.is_manual}')

    key_to_use = args.key



    if args.data.endswith(".mat"):
        print("solo file")
        frame = loadmat(args.data)
        frame = frame["TwoD_spectr_UP2"]
        frame = np.array(frame)
        pointcloud_coord, pointcloud_color = pointcloud_coords_generation(frame, threshold = args.threshold)
        sample_visualization(pointcloud_coord)
    else:
        #return frames func (directory, num_frames, por, key, vis_time, size) -> frames np ndarray
        frames = []
        filenames_list = glob.glob(os.path.join(args.data, args.filter))
        num_frames = args.num
        try:
            num_frames = int(num_frames)
        except:
            num_frames = len(filenames_list)

        if num_frames > len(filenames_list):
            num_frames = len(filenames_list)

        for filename in filenames_list[0:num_frames]: # hope here all files are sorted
            print(filename)
            start_time = time.clock()
            frame = loadmat(filename)
            frame = frame["TwoD_spectr_UP2"] #what key?!
            frames.append(frame)
            print("time opening:", time.clock() - start_time)
            # start_time = time.clock()
            # __ = pointcloud_coords_generation(frame)
            # print("time creating pcd:", time.clock() - start_time)
        frames = np.array(frames)

        if not(args.is_manual):
            animateFramesNative(frames, directory_output= args.output, threshold = args.threshold, load_viewpoint = args.is_viewpoint)
        else:
            custom_draw_geometry_with_key_callback(frames, threshold = args.threshold, is_loop = args.loop_mode)

        if args.output:
            if args.type == "gif":
                make_gif(directory = args.output)
            elif args.type == "avi":
                make_video(directory= args.output)
            else:
                print("sorry, can't find this type of output, use gif or avi")
