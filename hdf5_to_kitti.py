#!/home/luozhen/anaconda3/envs/weathernet/bin/python3.6
import numpy as np
import h5py
import glob
import os

PATH = "/media/luozhen/Blurryface SSD/数据集/点云语义分割/雨雾天气/cnn_denoising/test_01/"
FILE_PATH = "/media/luozhen/Blurryface SSD/数据集/点云语义分割/雨雾天气/hdf5_to_kitti/dataset/sequences/02/velodyne"
LABEL_PATH = "/media/luozhen/Blurryface SSD/数据集/点云语义分割/雨雾天气/hdf5_to_kitti/dataset/sequences/02/labels"


class Hdf5ToKitti:
    def __init__(self):
        # define hdf5 data format
        self.channels = ['labels_1', 'distance_m_1', 'intensity_1', 'sensorX_1', 'sensorY_1', 'sensorZ_1']
        self.sensorX_1 = None
        self.sensorY_1 = None
        self.sensorZ_1 = None
        self.distance_m_1 = None
        self.intensity_1 = None
        self.labels_1 = None
        self.point_size = None
        self.not_zero = None

    def load_hdf5_file(self, filename):
        """load one single hdf5 file with point cloud data
        e.g. sensorX_1 contains the x-coordinates in a projected 32x400 view
        """
        with h5py.File(filename, "r", driver='core') as hdf5:
            self.sensorX_1 = hdf5.get('sensorX_1')[()]
            self.sensorY_1 = hdf5.get('sensorY_1')[()]
            self.sensorZ_1 = hdf5.get('sensorZ_1')[()]
            self.distance_m_1 = hdf5.get('distance_m_1')[()]
            self.intensity_1 = hdf5.get('intensity_1')[()]
            self.labels_1 = hdf5.get('labels_1')[()]
            self.point_size = self.sensorX_1.shape[0] * self.sensorX_1.shape[1]
            self.not_zero = np.where(self.distance_m_1.reshape(-1, 1) != 0)

    def trans_point_cloud(self, path, filename):
        points = np.concatenate((self.sensorX_1.reshape(-1, 1)[self.not_zero].reshape(-1, 1),
                                 self.sensorY_1.reshape(-1, 1)[self.not_zero].reshape(-1, 1),
                                 self.sensorZ_1.reshape(-1, 1)[self.not_zero].reshape(-1, 1),
                                 self.intensity_1.reshape(-1, 1)[self.not_zero].reshape(-1, 1)), axis=1)
        filename = str(filename)
        filename = filename.zfill(6)
        points.astype('float32').tofile(os.path.join(path, filename) + '.bin')

    def trans_label(self, path, filename):
        # create uint16 kitti labels - instances all 0, not given
        semantic_ids = np.uint16(self.labels_1.reshape(-1, 1)[self.not_zero])
        instance_ids = np.zeros(len(semantic_ids), dtype=np.uint16)
        labels = np.empty(0, dtype=np.int32)
        if np.size(semantic_ids) == np.size(instance_ids):
            for semantic_id, instance_id in zip(semantic_ids, instance_ids):
                labels = np.append(labels, (instance_id << 16) + semantic_id)
            # Write combined label
            filename = str(filename)
            filename = filename.zfill(6)
            labels.astype('int32').tofile(os.path.join(path, filename) + '.label')
        else:
            print("Instance and Semantic ID not consistent, different array size!")
            return


if __name__ == "__main__":
    transfer = Hdf5ToKitti()
    files = sorted(glob.glob(PATH + '*/*.hdf5'))
    print('Directory {} contains are {} hdf5-files'.format(PATH, len(files)))

    if len(files) == 0:
        print('Please check the input dir {}. Could not find any hdf5-file'.format(PATH))
    else:
        print('Start transfer data')
        for frame, file in enumerate(files):
            print('{:04d} / {} file_name:{}'.format(frame, len(files), file))

            # load file
            transfer.load_hdf5_file(file)

            # transfer point cloud to .bin
            transfer.trans_point_cloud(FILE_PATH, frame)

            # transfer point label to .label
            transfer.trans_label(LABEL_PATH, frame)

    print('*' * 80)
    print('End of Transfer')
