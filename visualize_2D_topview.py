import os
import json
import csv
import math
from collections import defaultdict
import numpy as np
import cv2


class CoordMapper:
    def __init__(self, camera_config_file):
        self._camera_parameters = defaultdict(dict)
        self._camera_configs = json.load(open(os.path.join(camera_config_file)))
        self._camera_ids = list()

        for camera_param in self._camera_configs['Cameras']:
            cam_id = camera_param['CameraId']
            self._camera_ids.append(cam_id)

            self._camera_parameters[cam_id]['Translation'] = np.asarray(
                camera_param['ExtrinsicParameters']['Translation'])
            self._camera_parameters[cam_id]['Rotation'] = np.asarray(
                camera_param['ExtrinsicParameters']['Rotation']).reshape((3, 3))

            self._camera_parameters[cam_id]['FInv'] = np.asarray([
                1 / camera_param['IntrinsicParameters']['Fx'],
                1 / camera_param['IntrinsicParameters']['Fy'], 1
            ])
            self._camera_parameters[cam_id]['C'] = np.asarray([
                camera_param['IntrinsicParameters']['Cx'],
                camera_param['IntrinsicParameters']['Cy'], 0
            ])

        self._discretization_factorX = 1.0 / (
                (self._camera_configs['Space']['MaxU'] - self._camera_configs['Space']['MinU']) / (math.floor(
            (self._camera_configs['Space']['MaxU'] - self._camera_configs['Space']['MinU']) /
            self._camera_configs['Space']['VoxelSizeInMM']) - 1))
        self._discretization_factorY = 1.0 / (
                (self._camera_configs['Space']['MaxV'] - self._camera_configs['Space']['MinV']) / (math.floor(
            (self._camera_configs['Space']['MaxV'] - self._camera_configs['Space']['MinV']) /
            self._camera_configs['Space']['VoxelSizeInMM']) - 1))
        self._discretization_factor = np.asarray([self._discretization_factorX, self._discretization_factorY, 1])

        self._min_volume = np.asarray([
            self._camera_configs['Space']['MinU'], self._camera_configs['Space']['MinV'],
            self._camera_configs['Space']['MinW']
        ])

    def projection(self, person_center, camera_id):
        topdown_coords = np.transpose(
            np.asarray([[person_center['X'], person_center['Y'], 0]]))
        world_coord = topdown_coords / self._discretization_factor[:, np.newaxis] + self._min_volume[:, np.newaxis]
        uvw = np.linalg.inv(self._camera_parameters[camera_id]['Rotation']) @ (
                world_coord - self._camera_parameters[camera_id]['Translation'][:, np.newaxis])
        pixel_coords = uvw / self._camera_parameters[camera_id]['FInv'][:, np.newaxis] / uvw[
                                                                                         2, :] + \
                       self._camera_parameters[camera_id]['C'][:, np.newaxis]
        return pixel_coords[0][0], pixel_coords[1][0]


class DataReader:
    def __init__(self, image_dir, label_dir, num_cam):
        self.label_dir = label_dir
        self._image_dir = image_dir
        self._num_cam = num_cam
        self._mapper = CoordMapper('train/calibrations/cafe_shop/calibrations.json')

        # Tạo thư mục output nếu chưa tồn tại
        os.makedirs('output', exist_ok=True)

        # Khởi tạo video writers
        self._video_writers = {}
        self._topview_writer = None

        # Tính toán kích thước và tỷ lệ cho top-view
        self._min_U = self._mapper._camera_configs['Space']['MinU']
        self._max_U = self._mapper._camera_configs['Space']['MaxU']
        self._min_V = self._mapper._camera_configs['Space']['MinV']
        self._max_V = self._mapper._camera_configs['Space']['MaxV']
        range_U = self._max_U - self._min_U
        range_V = self._max_V - self._min_V
        if range_U > range_V:
            self._scale = 800 / range_U
            self._topview_width = 800
            self._topview_height = int(range_V * self._scale)
        else:
            self._scale = 800 / range_V
            self._topview_height = 800
            self._topview_width = int(range_U * self._scale)

    def read(self, frame_id):
        tracklets = []
        with open(os.path.join(self.label_dir, 'topdown_' + str(frame_id).zfill(5) + '.csv'), 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                tracklets.append(
                    {'center_coordinate': np.asarray([int(float(row[2])), int(float(row[1]))]), 'id': int(row[0])})
        rgbs = {}
        for cam_id in range(1, self._num_cam + 1):
            rgbs[cam_id] = cv2.imread(
                os.path.join(self._image_dir, 'rgb_' + str(frame_id).zfill(5) + '_' + str(cam_id) + '.jpg'),
                cv2.IMREAD_UNCHANGED)

        self.plot(tracklets, rgbs, frame_id)
        return rgbs, tracklets

    def plot(self, tracklets, rgbs, frame_id):
        # Khởi tạo video writers nếu chưa có
        if not self._video_writers:
            for cam_id in range(1, self._num_cam + 1):
                height, width = rgbs[cam_id].shape[:2]
                self._video_writers[cam_id] = cv2.VideoWriter(
                    os.path.join('output', f'camera_view_{cam_id}.avi'),
                    cv2.VideoWriter_fourcc(*'XVID'),
                    30.0,  # Giả định 30 fps, có thể điều chỉnh
                    (width, height)
                )

        if self._topview_writer is None:
            self._topview_writer = cv2.VideoWriter(
                os.path.join('output', 'topview.avi'),
                cv2.VideoWriter_fourcc(*'XVID'),
                30.0,
                (self._topview_width, self._topview_height)
            )

        # Xử lý từng chế độ xem camera
        for cam_id in range(1, self._num_cam + 1):
            rgb = rgbs[cam_id].copy()  # Sao chép để tránh sửa đổi hình gốc
            for tracklet in tracklets:
                x, y = self._mapper.projection(
                    {'X': tracklet['center_coordinate'][0], 'Y': tracklet['center_coordinate'][1]}, cam_id)
                cv2.circle(rgb, (int(x), int(y)), 5, (0, 0, 255), 2)
            self._video_writers[cam_id].write(rgb)

        # Tạo hình ảnh top-view
        topview_img = np.zeros((self._topview_height, self._topview_width, 3), dtype=np.uint8)
        dfX, dfY = self._mapper._discretization_factor[:2]
        for tracklet in tracklets:
            X, Y = tracklet['center_coordinate']
            U_world = X / dfX + self._min_U
            V_world = Y / dfY + self._min_V
            x = int((U_world - self._min_U) * self._scale)
            y = int((V_world - self._min_V) * self._scale)
            cv2.circle(topview_img, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(topview_img, str(tracklet['id']), (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        self._topview_writer.write(topview_img)

    def finish(self):
        # Giải phóng tất cả video writers
        for writer in self._video_writers.values():
            writer.release()
        if self._topview_writer is not None:
            self._topview_writer.release()

def combine_videos():
    """
    Hàm này kết hợp năm video thành một video duy nhất, hiển thị tất cả cùng lúc.
    Bốn camera view được sắp xếp trong lưới 2x2, và topview được đặt bên phải.
    """
    # Mở các tệp video
    cap1 = cv2.VideoCapture('output/camera_view_1.avi')
    cap2 = cv2.VideoCapture('output/camera_view_2.avi')
    cap3 = cv2.VideoCapture('output/camera_view_3.avi')
    cap4 = cv2.VideoCapture('output/camera_view_4.avi')
    cap_top = cv2.VideoCapture('output/topview.avi')

    # Lấy thuộc tính của video (giả sử các camera view có cùng kích thước)
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    width_top = int(cap_top.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_top = int(cap_top.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Định nghĩa kích thước của video kết hợp
    combined_width = 3 * width1  # Chiều rộng gấp 3 lần camera view
    combined_height = 2 * height1  # Chiều cao gấp 2 lần camera view

    # Tạo VideoWriter cho video đầu ra
    out = cv2.VideoWriter('output/combined.avi', cv2.VideoWriter_fourcc(*'XVID'),
                         fps1, (combined_width, combined_height))

    while True:
        # Đọc khung hình từ mỗi video
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()
        ret4, frame4 = cap4.read()
        ret_top, frame_top = cap_top.read()

        # Nếu bất kỳ video nào kết thúc, thoát vòng lặp
        if not (ret1 and ret2 and ret3 and ret4 and ret_top):
            break

        # Tạo khung hình kết hợp, khởi tạo với màu đen
        combined = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

        # Sắp xếp bốn camera view trong lưới 2x2 ở bên trái
        combined[0:height1, 0:width1] = frame1  # Góc trên-trái
        combined[0:height1, width1:2*width1] = frame2  # Góc trên-phải
        combined[height1:2*height1, 0:width1] = frame3  # Góc dưới-trái
        combined[height1:2*height1, width1:2*width1] = frame4  # Góc dưới-phải

        # Xử lý topview: thay đổi kích thước để vừa với khu vực bên phải
        if width_top > 0 and height_top > 0:
            # Tính tỷ lệ thu phóng để topview vừa với width1 x (2*height1) mà không bị méo
            scale = min(width1 / width_top, 2 * height1 / height_top)
            new_width = int(width_top * scale)
            new_height = int(height_top * scale)
            if new_width > 0 and new_height > 0:
                # Thay đổi kích thước topview
                resized_topview = cv2.resize(frame_top, (new_width, new_height))
                # Tính vị trí để căn giữa topview trong khu vực bên phải
                start_x = 2 * width1 + (width1 - new_width) // 2
                start_y = (2 * height1 - new_height) // 2
                # Chép topview đã thay đổi kích thước vào khung hình kết hợp
                combined[start_y:start_y + new_height, start_x:start_x + new_width] = resized_topview

        # Ghi khung hình kết hợp vào video đầu ra
        out.write(combined)

    # Giải phóng tài nguyên
    cap1.release()
    cap2.release()
    cap3.release()
    cap4.release()
    cap_top.release()
    out.release()
    print("Đã tạo video kết hợp tại 'output/combined.avi'")


if __name__ == '__main__':
    image_dir = 'train/images/63am/cafe_shop_0'
    topdown_label_dir = 'train/topdown_labels/63am/cafe_shop_0'
    num_cam = 4
    data_reader = DataReader(image_dir, topdown_label_dir, num_cam)

    start_frame = 0
    end_frame = 100  # Điều chỉnh số lượng khung hình cần xử lý
    for frame_id in range(start_frame, end_frame):
        data_reader.read(frame_id)
    data_reader.finish()

    # Gọi hàm để tạo video kết hợp
    combine_videos()