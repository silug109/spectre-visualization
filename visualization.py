import cv2
import imageio
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import open3d as o3d

from data_util import *


# 3D visualization

def pointcloud_coords_generation(frame, range_max = 67, azimuth_range_max = 57, elevation_max = 16, por = 0.5):
    '''
    function to transform tensor represantation(X_shape, Y_shape, Z_shape) to coordinate representation (N_points, 3)
    :param frame: ndarray
    :param range_max: max distance in absolute coordinates to which transform pointclouds
    :param azimuth_range_max: max azimuth value in absolute coordinates
    :param elevation_max: max elevation angle
    :param threshold: threshold with which filter all points in grid of tensor representation
    :return: ndarray[N,4] - coordinates, ndarray[N,3] - color
    '''

    R = np.arange(0, range_max, range_max / frame.shape[0])
    theta = np.arange(-azimuth_range_max, azimuth_range_max, 2 * azimuth_range_max / frame.shape[1])
    epsilon = np.arange(0, elevation_max, elevation_max / frame.shape[2])

    theta_sin = np.sin(theta * np.pi / 180)
    theta_cos = np.cos(theta * np.pi / 180)
    epsilon_sin = np.sin(epsilon * np.pi / 180)
    epsilon_cos = np.cos(epsilon * np.pi / 180)

    tup_coord = np.nonzero(frame > threshold)

    x = np.expand_dims((R[tup_coord[0]] * theta_cos[tup_coord[1]] * epsilon_cos[tup_coord[2]]), 1)
    y = np.expand_dims((R[tup_coord[0]] * theta_sin[tup_coord[1]] * epsilon_cos[tup_coord[2]]), 1)
    z = np.expand_dims((R[tup_coord[0]] * epsilon_sin[tup_coord[2]]), 1)

    points = np.concatenate((x, y, z, np.expand_dims(frame[tup_coord], 1)), axis=1)

    points_cord = np.array(points)
    colors_arr = np.swapaxes(np.vstack((points_cord[:, 3], points_cord[:, 3], points_cord[:, 3])) / 255, 0, 1)
    return points_cord, colors_arr

def spectr_generation_floor(spectr, value_of_increase=4):
    '''
    Function for ground drawing.

    :param spectr: 2d images that we want to draw on floor
    :param value_of_increase: number of upsampling
    :return: ndarray[N,4] - coordinates, ndarray[N,3] - colors
    '''
    new_spectr = cv2.resize(spectr, dsize=(spectr.shape[0] * value_of_increase, spectr.shape[1] * value_of_increase),
                            interpolation=cv2.INTER_CUBIC)
    kernel = np.ones((5, 5))
    new_spectr = cv2.dilate(new_spectr, kernel, iterations=4)

    R = np.arange(0, range_max, range_max / frame.shape[0])
    theta = np.arange(-azimuth_range_max, azimuth_range_max, 2 * azimuth_range_max / frame.shape[1])

    theta_sin = np.sin(theta * np.pi / 180)
    theta_cos = np.cos(theta * np.pi / 180)

    tup_coord = np.nonzero(frame > threshold)

    x = np.expand_dims((R[tup_coord[0]] * theta_cos[tup_coord[1]]),1)
    y = np.expand_dims((R[tup_coord[0]] * theta_sin[tup_coord[1]]),1)
    z = np.expand_dims(np.zeros(len(tup_coord[0])) - 0.5 ,1)

    points = np.concatenate((x, y, z, np.expand_dims(frame[tup_coord], 1)), axis=1)

    points_cord = np.array(points)
    colors_arr = np.swapaxes(np.vstack((points_cord[:, 3], points_cord[:, 3], points_cord[:, 3])) / 255, 0, 1)
    colors_arr = 1 - colors_arr
    return points_cord, colors_arr

# def pointcloud_coords_generation(frame, range_max = 67, azimuth_range_max = 57, elevation_max = 16):
#     '''
#     :param frame: (config.size[1], size[2], config.size[3])
#     :return: ndarray(num_points, 4)
#     '''
#     R = np.arange(0,range_max, range_max/frame.shape[0])
#     theta = np.arange(-azimuth_range_max, azimuth_range_max, 2*azimuth_range_max/frame.shape[1])
#     epsilon = np.arange(0,elevation_max,elevation_max/frame.shape[2])
#
#     points_cord = []
#     for i in range(frame.shape[0]):
#         for j in range(frame.shape[1]):
#             for k in range(0,frame.shape[2]-6):
#                 if frame[i,j,k] > 0.1:
#
#                     x = R[i]* np.cos(theta[j]*np.pi/180)* np.cos(epsilon[k]*np.pi/180)
#                     y = R[i]* np.sin(theta[j]*np.pi/180)* np.cos(epsilon[k]*np.pi/180)
#                     z = R[i]* np.sin(epsilon[k]*np.pi/180)
#
#                     points_cord.append([x,y,z,frame[i,j,k]])
#
#     points_cord = np.array(points_cord)
#     colors_arr  = np.swapaxes(np.vstack((points_cord[:,3],points_cord[:,3],points_cord[:,3]))/255,0,1)
#     return points_cord , colors_arr

def create_pointcloud(points_coord, colors_arr = None):
    '''
    Create pointcloud object from point coordinate
    :param points_coord: [N,3/4] ndarray, where N - quantity of points to draw
    :param colors_arr: [N, 3] ndaaray
    :return: open3d.geometry.Pointcloud object
    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_coord[:, :3])
    if colors_arr != None:
        pcd.colors = o3d.utility.Vector3dVector()
    return pcd


#some objects
def line_creation(R = 60, visualize = False):
    '''
        create radial lines in 0XY to show radiuses in spherical coordinates
        :param R: distance until which create lines
        :param visualize: Show or not
        :return: open3d.geometry.Lineset object
        '''
    points = [[0,0,0]] + [[R*np.cos(azi*np.pi/180), R*np.sin(azi*np.pi/180), 0] for azi in range(-90,91,10)]
    lines = [[0,i] for i in range(1, len(points))]
    colors = [[0, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    if visualize:
        o3d.visualization.draw_geometries([line_set])
    return line_set

def conc_creation(R = 60):
    '''
    create concetric lines in 0XY to show universal distances in spherical coordinates
    :param R: distance until which create lines
    :return: open3d.geometry.Lineset object

    usage conc_obj = [conc_creation(R) for R in range(0,61,10)]

    '''
    points = [[R*np.cos(azi*np.pi/180), R*np.sin(azi*np.pi/180), 0] for azi in range(-90,91,1)]
    lines = [[i,i+1] for i in range(1,len(points))]
    colors = [[0, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set


#visualization functions

def sample_visualization(points_coord):
    '''
    simple function to visualize point coordinates
    :param points_coord: ndarray [N,3/4]
    '''
    pcd = create_pointcloud(points_coord)
    o3d.visualization.draw_geometries([pcd])
    return 'success'

def custom_draw_geometry_with_key_callback(frames):
    """
        Function to visualize list of frames and control proccess through keyboard.
        :param frames:
        """
    global counter
    counter = 0

    frame = frames[counter]

    start_time = time.time()
    points_cord, colors_arr = pointcloud_coords_generation(frame)
    print("pcd:", time.time() - start_time)
    pcd = create_pointcloud(points_cord)

    pcd.translate(np.array([0, 0, 2]))

    line_set = line_creation(60)
    conc_obj = [conc_creation(R) for R in range(1, 61, 10)]

    spectr_pcd = o3d.geometry.PointCloud()

    twod_spectr = np.sum(frame, axis=-1)
    twod_spectr = 255 * (twod_spectr - np.min(twod_spectr)) / (np.max(twod_spectr) - np.min(twod_spectr))

    start_time = time.time()
    points, colors = spectr_generation(twod_spectr, value_of_increase=2)
    print("back", time.time() - start_time)

    spectr_pcd.points = o3d.utility.Vector3dVector(points[..., :3])
    spectr_pcd.colors = o3d.utility.Vector3dVector(colors)

    def save_render_option(vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters('viewpoint.json', param)
        return False

    def load_render_option(vis):
        ctr = vis.get_view_control()
        param = o3d.io.read_pinhole_camera_parameters('viewpoint.json')
        ctr.convert_from_pinhole_camera_parameters(param)
        return False

    def change_frame(vis):
        ctr = vis.get_view_control()
        fov = ctr.get_field_of_view()

        num_frame = np.random.randint(0, 40)
        new_pcd_attr, new_colors_arr = pointcloud_coords_generation(load_from_matfile(matr, num_frame))
        new_pcd_cords = new_pcd_attr[0][:, :3]
        pcd.points = o3d.utility.Vector3dVector(new_pcd_cords)
        #         pcd.colors = o3d.utility.Vector3dVector(new_pcd_colors)
        vis.update_geometry()

        return False

    def next_frame(vis):
        global counter

        ctr = vis.get_view_control()
        fov = ctr.get_field_of_view()

        #         vis.capture_screen_image("images2gif_car/new_"+ str(counter) + '.png')
        vis.capture_screen_image(direct + "/" + filename_ext + "_" + str(counter) + '.png')

        counter += 1
        num_frame = counter

        try:
            new_frame = frames[num_frame]
        except IndexError:
            counter = 0
            num_frame = counter
            new_frame = frames[num_frame]

        frame = new_frame
        start_time = time.time()
        new_pcd_cords, new_pcd_colors = pointcloud_coords_generation(frame)
        print("pcd:", time.time() - start_time)

        pcd.points = o3d.utility.Vector3dVector(new_pcd_cords[..., :3])
        #         pcd.colors = o3d.utility.Vector3dVector(new_colors)

        pcd.translate(np.array([0, 0, 2]))

        twod_spectr = np.sum(frame, axis=-1)
        twod_spectr = 255 * (twod_spectr - np.min(twod_spectr)) / (np.max(twod_spectr) - np.min(twod_spectr))

        start_time = time.time()
        points, colors = spectr_generation(twod_spectr, value_of_increase=2)
        print("back:", time.time() - start_time)

        spectr_pcd.points = o3d.utility.Vector3dVector(points[..., :3])
        spectr_pcd.colors = o3d.utility.Vector3dVector(colors)

        vis.update_geometry()

        print(counter, num_frame)

        return False

    key_to_callback = {}
    key_to_callback[ord("Z")] = change_frame
    key_to_callback[ord("X")] = next_frame
    key_to_callback[ord("A")] = load_render_option
    key_to_callback[ord("B")] = save_render_option
    o3d.visualization.draw_geometries_with_key_callbacks([line_set, spectr_pcd, pcd, *conc_obj], key_to_callback)

def animateFrames(frames, directory_output = None):
    '''
    visualize frames each by each
    '''
    global counter
    counter = 0
    frame = frames[counter]
    coords,_ = pointcloud_coords_generation(frame)
    pcd = create_pointcloud(coords)

    if not(directory_output is None) and not(os.path.exists(directory_output)):
        print(f"create dir {directory_output}")
        os.mkdir(directory_output)

    def change_frame(vis):
        global counter
        # print(f"changing to {counter}")

        if os.path.exists("viewpoint.json"):
            ctr = vis.get_view_control()
            param = o3d.io.read_pinhole_camera_parameters('viewpoint.json')
            ctr.convert_from_pinhole_camera_parameters(param)

        if not(directory_output is None):
            vis.capture_screen_image(directory_output + "/" + str(counter) + '.png')

        counter += 1
        try:
            frame = frames[counter]
        except:
            counter = 0
            frame = frames[counter]

        coords, _ = pointcloud_coords_generation(frame)

        pcd.points = o3d.utility.Vector3dVector(coords[:,:3])
        vis.update_geometry()
        return False

    # native function and realization
    o3d.visualization.draw_geometries_with_animation_callback([pcd], change_frame)

def animateFramesNative(frames, directory_output = None):
    '''
    Function that visualize frames each-by-each. load viewpoint from file viewpoint.json is exist in root.
    Stops when window is closed. Written through more flexible API.
    :param frames: ndarray(N,one_frame_shape)
    :param directory_output: if not None save screen as image to directory_output. If None - don't save
    '''

    vis = o3d.visualization.Visualizer()
    vis.create_window("visualization animation")
    ctr = vis.get_view_control()

    if os.path.exists("viewpoint.json"):
        param = o3d.io.read_pinhole_camera_parameters('viewpoint.json')
        ctr.convert_from_pinhole_camera_parameters(param)

    if not(directory_output is None) and not(os.path.exists(directory_output)):
        print(f"create dir {directory_output}")
        os.mkdir(directory_output)

    counter = 0

    frame = frames[counter]

    pointcoords,_ = pointcloud_coords_generation(frame)
    pcd = create_pointcloud(pointcoords)

    vis.add_geometry(pcd)

    while True:

        if not(directory_output is None):
            vis.capture_screen_image(directory_output + "/" + str(counter) + '.png')

        counter += 1
        try:
            frame = frames[counter]
        except:
            counter = 0
            frame = frames[counter]

        pointcoords, _ = pointcloud_coords_generation(frame)
        pcd.points = o3d.utility.Vector3dVector(pointcoords[...,:3])
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()

        if not vis.poll_events():
            break


#utilities
def save_as_bin(points_cord, filename = 'something.bin'):
    '''
    Function allows to save pointclouds coordinates to bin file
    Proposed to save test coordinates or for future. Needed to remember shape
    :param points_cord: np.ndarray (N,4/3)
    '''
    point_array = points_cord.astype(np.float32)
    point_array = point_array.reshape((-1))
    point_array.tofile(filename)
    return 'Saved'

def make_gif(directory, need_sort = True):
    '''
    create gif out of list of .png images in directory
    '''
    images_listpath = glob.glob(directory+'*.png')
    if need_sort:
        right_order = sorted(images_listpath, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
        # sort filenames by int if needed
    else:
        right_order = images_listpath
    images = []
    for filename in right_order:
        images.append(imageio.imread(filename))
    imageio.mimsave(directory + '/movie.gif', images)
    return 'Success'

def make_video(directory):
    pass
    list_spectr_frames = glob.glob(directory + "/*.png")
    # list_spectr_frames = sorted(list_spectr_frames, key=lambda x: int(x.split("_")[-1][:-4]))
    print("found frames with spectr: ",len(list_spectr_frames))

    out_size = ()

    fft_spectr = cv2.imread(list_spectr_frames[0])
    height, width, layers = fft_spectr.shape

    out = cv2.VideoWriter(directory + '/movie.avi', cv2.VideoWriter_fourcc(*'DIVX'), 25, (width, height))
    for ind in range(len(list_spectr_frames)):
        fft_spectr = cv2.imread(list_spectr_frames[ind])
        height, width, layers = fft_spectr.shape
        spectr_size = (height, width)

        output = np.ones(shape=(height, width, 3), dtype=np.uint8)
        output = fft_spectr
        for i in range(10):
            out.write(output)
    out.release()

def visualise_bird_view(frame ,targets, boxes, do_nms = False, do_photo = False, nms = False):
    '''

    :param frame:
    :param targets:
    :param boxes:
    :param do_nms:
    :param do_photo:
    :param nms:
    :return:
    '''


    fig, axs = plt.subplots(1 ,2, figsize = (10 ,10))

    im = np.sum(frame ,axis = (0 ,3 ,4))

    if nms == True:
        boxes = do_nms_exp(boxes, 0.05)
        print(len(boxes))

    axs[0].imshow(im)
    axs[1].imshow(im)
    x_ticks = np.arange(0, im.shape[1], 20)
    y_ticks = np.arange(0, im.shape[0], 20)
    axs[0].set_xticks(x_ticks)
    axs[0].set_yticks(y_ticks)
    axs[1].set_xticks(x_ticks)
    axs[1].set_yticks(y_ticks)
    axs[0].set_title('target boxes')
    axs[1].set_title('predicted boxes')

    for num_target in range(targets.shape[0]):
        x = targets[num_target ,1]
        y = targets[num_target ,2]
        h = targets[num_target ,4]
        w = targets[num_target ,5]
        x_t = x- h / 2
        x_b = x + h / 2
        y_t = y - w / 2
        y_b = y + w / 2
        rect = patches.Rectangle((y_t, x_t), w, h, linewidth=1, edgecolor='r', facecolor='none')
        axs[0].add_patch(rect)

    for box in boxes:
        x = box[0]
        y = box[1]
        h = box[3]
        w = box[4]
        x_t = x - h / 2
        x_b = x + h / 2
        y_t = y - w / 2
        y_b = y + w / 2
        class_box = box[7]
        rect = patches.Rectangle((y_t, x_t), w, h, linewidth=1, edgecolor= color_dict[class_box], facecolor='none')
        axs[1].add_patch(rect)

    if do_photo:
        plt.savefig('comparison.png')

def show_comparison_object_level(y_true, y_pred, save = False):
    fig, axs = plt.subplots(1, 3)
    fig.suptitle('confidence map')
    axs[0].imshow(np.sum(y_true, axis=(0, 3, 4, 5)))
    axs[0].set_title('true map')
    axs[1].imshow(y_true[0, :, :, 2, 0, N_DIM*2])
    axs[2].set_title('pred map')
    axs[2].imshow(sigmoid(y_pred[0, :, :, 2, 0, N_DIM*2]))

    if save:
        plt.savefig('zhopa.jpg')

def erosion_frame(frame):
    kernel = np.ones((7, 7), np.uint8)
    erosion = cv2.dilate(frame[0, ..., 0], kernel, iterations=1)
    return erosion

def histogram_equalization(frame):
    '''
    function returns 3d frame with equalised histogram
    :param frame: numpy tensor of shape (1,GRID_H, GRID_W, GRID_D, num_anchors, num_classes+4/6/+1)
    :return: same shape frame
    '''

    histogram_array = np.histogram(frame[0, ..., 0], bins=256)
    cdf = histogram_array[0].cumsum()
    cdf_normalized = cdf * histogram_array[0].max() / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    frame_equalised = cdf[frame[0, ..., 0]]
    return frame_equalised

# def make_video_many_sources(directory):
#     pass
#     list_spectr_frames = glob.glob(directory + "/*.png")
#     # list_spectr_frames = sorted(list_spectr_frames, key=lambda x: int(x.split("_")[-1][:-4]))
#     print("found frames with spectr: ",len(list_spectr_frames))
#     #
#     # list_ib_frames = glob.glob(direct + "/IB_*.png")
#     # list_ib_frames = sorted(list_ib_frames, key=lambda x: int(x.split("_")[-1][:-4]))
#     # print(len(list_ib_frames), list_ib_frames)
#     #
#     # list_video_frames = glob.glob("images/VIDEOC*.png")
#     # list_video_frames = sorted(list_video_frames, key=lambda x: int(x.split("_")[-1][:-4]))
#     # print(len(list_video_frames), list_video_frames)
#
#     out_size = ()
#
#     out = cv2.VideoWriter(directory + '/movie.avi', cv2.VideoWriter_fourcc(*'DIVX'), 25, (2560, 1440))
#     for ind in range(len(list_spectr_frames)):
#         fft_spectr = cv2.imread(list_spectr_frames[ind])
#         height, width, layers = fft_spectr.shape
#         spectr_size = (height, width)
#
#     #
#     #     video = cv2.imread(list_video_frames[ind])
#     #     height, width, layers = frame.shape
#     #     video_size = (height, width)
#     #
#     #     ib = cv2.imread(list_ib_frames[ind])
#     #     height, width, layers = ib.shape
#     #     ib_size = (height, width)
#     #
#     #     # 2560,1440
#         output = np.ones(shape=(1440, 2560, 3), dtype=np.uint8)
#     #     output = np.ones(shape=(height, width, 3), dtype = np.uint8)
#         output[:height, :width,:] = fft_spectr
#
#     #     fft_spectr = cv2.resize(fft_spectr, dsize=(int(2560 / 2), int(1440 / 2)), interpolation=cv2.INTER_CUBIC)
#     #     ib = cv2.resize(ib, dsize=(int(2560 / 2), int(1440 / 2)), interpolation=cv2.INTER_CUBIC)
#     #     video = cv2.resize(video, dsize=(960, 720), interpolation=cv2.INTER_CUBIC)
#     #     print(fft_spectr.shape)
#     #
#     #     output[0:int(1440 / 2), 0:int(2560 / 2)] = fft_spectr.astype(np.uint8)
#     #     output[int(1440 / 2):1440, 0:int(2560 / 2)] = ib.astype(np.uint8)
#     #     output[1440 // 2 - video.shape[0] // 2: 1440 // 2 - video.shape[0] // 2 + video.shape[0],
#     #     2560 // 2 + 2560 // 4 - video.shape[1] // 2:2560 // 2 + 2560 // 4 - video.shape[1] // 2 + video.shape[
#     #         1]] = video.astype(np.uint8)
#     #
#         for i in range(10):
#             out.write(output)
#     out.release()

# def labels(frame):
