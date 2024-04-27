from plyfile import PlyData
import numpy as np
import os
import progressbar
import shutil

parent_folder = "D:/Code/kitti_carla_simulator/KITTI_Dataset_CARLA_v0.9.15-B4-AVP3.0/Carla/Maps/SUSTech_COE_ParkingLot/generated/frames/"
output_path = "./"

index = 0
bar = progressbar.ProgressBar(max_value=len(os.listdir(parent_folder)))

velodyne_path = output_path + "velodyne/"
label_path = output_path + "labels/"
if os.path.exists(velodyne_path):
    shutil.rmtree(velodyne_path)
if os.path.exists(label_path):
    shutil.rmtree(label_path)
os.mkdir(velodyne_path)
os.mkdir(label_path)

for ply_name in os.listdir(parent_folder):
    ply_path = os.path.join(parent_folder, ply_name)
    cloud = PlyData.read(ply_path)
    meta = np.array(list(map(list, cloud.elements[0].data)))
    pos = meta[:, :4]
    label = meta[:, -1]

    pos = pos.astype(np.float32)
    label = label.astype(np.uint32)

    # remap label
    label[label == 1] = 40  # road - road
    label[label == 2] = 48  # sidewalk - sidewalk
    label[label == 3] = 50  # building - building
    label[label == 4] = 52  # wall - other-structure
    label[label == 5] = 51  # fence - fence
    label[label == 6] = 80  # pole - pole
    label[label == 7] = 99  # traffic light - other-object
    label[label == 8] = 81  # traffic sign - traffic-sign
    label[label == 9] = 70  # vegetation - vegetation
    label[label == 10] = 72 # terrain - terrain
    label[label == 11] = 0  # sky - unlabeled
    label[label == 12] = 30 # pedestrian - person
    label[label == 13] = 31 # rider - bicyclist
    label[label == 14] = 10 # car - car
    label[label == 15] = 18 # truck - truck
    label[label == 16] = 13 # bus - bus
    label[label == 17] = 16 # train - on-rails
    label[label == 18] = 15 # motorcycle - motorcycle
    label[label == 19] = 11 # bicycle - bicycle
    label[label == 20] = 20 # static - outlier
    label[label == 21] = 259    # dynamic - moving-other-vehicle
    label[label == 22] = 99 # other - other-object
    label[label == 23] = 49 # water - other-ground
    label[label == 24] = 60 # road line - lane-marking
    label[label == 25] = 49 # ground - other-ground
    label[label == 26] = 52  # bridge - other-structure
    label[label == 27] = 49 # rail - other-ground
    label[label == 28] = 51 # guard rail - fence
    label[label == 29] = 60 # lane-marking
    label[label == 30] = 44 # parking

    pos.tofile("velodyne/{:0>6}.bin".format(str(index)))
    label.tofile(("labels/{:0>6}.label").format(str(index)))

    index += 1
    bar.update(index)
bar.finish()

