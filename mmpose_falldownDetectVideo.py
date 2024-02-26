import mmcv
from mmcv import imread
import mmengine
from mmengine.registry import init_default_scope
import numpy as np
import os
import  cv2

os.chdir("/Users/jackylove/Coding/ML/pytorch/mmpose/mmpose")

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

local_runtime = False

try:
    from google.colab.patches import cv2_imshow  # for image visualization in colab
except:
    local_runtime = True


# 框（rectangle）可视化配置
bbox_color = (150, 0, 0) # 框的 BGR 颜色
bbox_color_fall = (0, 0, 255) # 框的 BGR 颜色
bbox_color_no_fall =(0,255,0)
bbox_thickness = 6                   # 框的线宽

# 框类别文字
bbox_labelstr = {
    'font_size':4,         # 字体大小
    'font_thickness':5,   # 字体粗细
    'offset_x':0,          # X 方向，文字偏移距离，向右为正
    'offset_y':-10,        # Y 方向，文字偏移距离，向下为正
}

img = 'tests/data/coco/000000197388.jpg'
# img = 'JC_codes/2361707043462_.pic_hd.jpg'
# img = 'JC_codes/RC46.jpg'

pose_config = 'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
det_config = 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
det_config = 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# device = 'cuda:0'
device = 'cpu'
cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))


# build detector
detector = init_detector(
    det_config,
    det_checkpoint,
    device=device
)


# build pose estimator
pose_estimator = init_pose_estimator(
    pose_config,
    pose_checkpoint,
    device=device,
    cfg_options=cfg_options
)

# init visualizer
pose_estimator.cfg.visualizer.radius = 3
pose_estimator.cfg.visualizer.line_width = 1
visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
# the dataset_meta is loaded from the checkpoint and
# then pass to the model in init_pose_estimator
visualizer.set_dataset_meta(pose_estimator.dataset_meta)


def visualize_img(img_path, detector, pose_estimator, visualizer,
                  show_interval, out_file):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    scope = detector.cfg.get('default_scope', 'mmdet')
    if scope is not None:
        init_default_scope(scope)
    detect_result = inference_detector(detector, img_path)
    pred_instance = detect_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                   pred_instance.scores > 0.3)]
    bboxes = bboxes[nms(bboxes, 0.3)][:, :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img_path, bboxes)
    data_samples = merge_data_samples(pose_results)

    # print(len(pose_results))
    # print(pose_results[0].pred_instances.keypoints)
    # print(pose_results[0].pred_instances.keypoint_scores)

    # show the results
    # img = mmcv.imread(img_path, channel_order='rgb')

    # visualizer.add_datasample(
    #     'result',
    #     img,
    #     data_sample=data_samples,
    #     draw_gt=False,
    #     draw_heatmap=True,
    #     draw_bbox=True,
    #     show=False,
    #     wait_time=show_interval,
    #     out_file=out_file,
    #     kpt_thr=0.3)
    return  pose_results

# pose_results = visualize_img(
#     img,
#     detector,
#     pose_estimator,
#     visualizer,
#     show_interval=0,
#     out_file=None)

# vis_result = visualizer.get_image()


def cosangle(x_A,y_A,x_B,y_B,x_C,y_C): #肩上亮点为 5，6   胯部 11，12   膝盖 13 14  脚 15，16


    # 点 A, B, C 的坐标
    A = np.array([x_A, y_A])
    B = np.array([x_B, y_B])
    C = np.array([x_C, y_C])

    # 向量 AB 和 BC
    AB = B - A
    BC = C - B

    # 计算点积
    dot_product = np.dot(AB, BC)

    # 计算向量AB和BC的模长
    magnitude_AB = np.linalg.norm(AB)
    magnitude_BC = np.linalg.norm(BC)

    # 计算夹角的余弦值
    cos_theta = dot_product / (magnitude_AB * magnitude_BC) #110

    # print(cos_theta)

    return cos_theta


def fall_or_not(pose_results):
    bbox_draw = []
    # print(pose_results[0].pred_instances.bboxes)
    for instance_id in range(len(pose_results)):
        bbox_instance ={}

        for keypoints in pose_results[instance_id].pred_instances.keypoints:

            print("how many times!")
            # print(keypoints)
            # for point_id in range(len(keypoints)):
            ponint5 = keypoints[5]
            ponint6 = keypoints[6]
            ponint11 = keypoints[11]
            ponint12 = keypoints[12]
            ponint13 = keypoints[13]
            ponint14 = keypoints[14]

            middle_56 = (ponint5[0]+ponint6[0])/2.0, (ponint5[1]+ponint6[1])/2.0
            middle_1112 =  (ponint11[0]+ponint12[0])/2.0, (ponint11[1]+ponint12[1])/2.0
            middle_1314 =  (ponint13[0]+ponint14[0])/2.0, (ponint13[1]+ponint14[1])/2.0

            cos_theta = cosangle(middle_56[0],middle_56[1],middle_1112[0],middle_1112[1],middle_1314[0],middle_1314[1])

            print(cos_theta)

        p1_x = pose_results[instance_id].pred_instances.bboxes[0][0]
        p1_y = pose_results[instance_id].pred_instances.bboxes[0][1]

        p2_x = pose_results[instance_id].pred_instances.bboxes[0][2]
        p2_y = pose_results[instance_id].pred_instances.bboxes[0][3]

        # middle_56[xyxy_num][1]>middle_1112[xyxy_num][1] 表示肩比胯低，又可能是 弯腰动作
        if middle_1314[1] < middle_56[1] or middle_1314[1] < middle_1112[1]:
            fall_flag = True


            print("Fall Down 1")
            #
            # img_fall = cv2.rectangle(image, (p1_x, p1_y), (p2_x, p2_y), bbox_color_fall,
            #                          bbox_thickness)
            #
            # img_fall = cv2.putText(img_fall, 'fall down',
            #                        (p1_x + bbox_labelstr['offset_x'], p1_y + bbox_labelstr['offset_y']),
            #                        cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color_fall,
            #                        bbox_labelstr['font_thickness'])


        # 框的大小 宽大于0.7高
        elif cos_theta > 0.9 and ((p2_x - p1_x) > 0.85 * (p2_y - p1_y)):
            print("Fall Down 2")
            fall_flag = True


            # img_fall = cv2.rectangle(image, (p1_x, p1_y), (p2_x, p2_y), bbox_color_fall,
            #                          bbox_thickness)
            #
            # img_fall = cv2.putText(img_fall, 'fall down',
            #                        (p1_x + bbox_labelstr['offset_x'], p1_y + bbox_labelstr['offset_y']),
            #                        cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color_fall,
            #                        bbox_labelstr['font_thickness'])
        else:
            print("ok")
            fall_flag = False


            # img_fall = cv2.rectangle(image, (p1_x, p1_y), (p2_x, p2_y), bbox_color,
            #                          bbox_thickness)
        if fall_flag:
            bbox_instance['fall_flag'] = True
        else:
            bbox_instance['fall_flag'] = False
        bbox_instance['bbox_xyxy'] = (p1_x,p1_y,p2_x,p2_y)
        bbox_draw.append(bbox_instance)

    print(bbox_draw)
    return bbox_draw





#show keypoints

# def points_visual():
#     src_img = cv2.imread(img)
#     for instance_id in range(len(pose_results)):
#
#         for keypoints in pose_results[instance_id].pred_instances.keypoints:
#             # print(keypoints)
#             for point in keypoints:
#                 # print(point[0])
#                 # print(type(point[0]))
#             # centre_point = keypoint.split(' ')
#                 cv2.circle(src_img, (int(point[0]),int(point[1])), 3, thickness=-10, color=(255,0,0))
#
#     cv2.imshow("demo",src_img)
#     cv2.waitKey(10000)
def points_visual2(img,bbox_draw):
    src_img = img
    # src_img = cv2.imread(img)

    for bbox in bbox_draw:
        p1_x = bbox['bbox_xyxy'][0]
        p1_y = bbox['bbox_xyxy'][1]
        p2_x = bbox['bbox_xyxy'][2]
        p2_y = bbox['bbox_xyxy'][3]
        print(p1_x,p1_y,p2_x,p2_y)
        if bbox['fall_flag']==True:
            img_fall = cv2.putText(src_img, 'fall down',
                                   (int(p1_x) + bbox_labelstr['offset_x'], int(p1_y) + bbox_labelstr['offset_y']),
                                   cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color_fall,
                                   bbox_labelstr['font_thickness'])
            img_fall = cv2.rectangle(src_img, (int(p1_x), int(p1_y)), (int(p2_x), int(p2_y)), bbox_color_fall,
                                 bbox_thickness)
        else:
            img_fall = cv2.rectangle(src_img, (int(p1_x), int(p1_y)), (int(p2_x), int(p2_y)), bbox_color_no_fall,
                                     bbox_thickness)


    return img_fall
    # cv2.imshow("demo",img_fall)
    # cv2.waitKey(10000)

# if pose_results is None or len(pose_results.)
# print(pose_results)




if __name__ == '__main__':
#     # created by Huang Lu
#     # 27/08/2016 17:24:55
#     # Department of EE, Tsinghua Univ.
#
    import cv2
    import numpy as np

    cap = cv2.VideoCapture("/Users/jackylove/Downloads/F4_project/test_pic_video/FallDown_TestVideos/Collapse-FalldownDetection-1.mp4")
    while (cap.isOpened()):
        # get a frame
        ret, frame = cap.read()
        pose_results = visualize_img(
                frame,
                detector,
                pose_estimator,
                visualizer,
                show_interval=0,
                out_file=None)
        bbox_draw = fall_or_not(pose_results)
        # points_visual()
        fall_frame =  points_visual2(frame,bbox_draw)

        # show a frame
        cv2.imshow("capture", fall_frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# if __name__ == '__main__':
#     import cv2
#
#     import numpy as np
#
#     videoname = 'videoname_out.mp4'
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     writer = cv2.VideoWriter(videoname, fourcc, 1.0, (1280, 960), True)
#     cap = cv2.VideoCapture("/Users/jackylove/Downloads/F4_project/test_pic_video/FallDown_TestVideos/FalldownDetection.mp4")
#     while (1):
#         # get a frame
#         ret, frame = cap.read()
#         pose_results = visualize_img(
#                 frame,
#                 detector,
#                 pose_estimator,
#                 visualizer,
#                 show_interval=0,
#                 out_file=None)
#         bbox_draw = fall_or_not(pose_results)
#         # points_visual()
#         fall_frame =  points_visual2(frame,bbox_draw)
#
#         # show a frame
#         # cv2.imshow("capture", fall_frame)
#         writer.write(fall_frame)
#         print("write frame!!")
#         if cv2.waitKey(100) & 0xFF == ord('q'):
#             break
#     writer.release()
#     cap.release()
#     cv2.destroyAllWindows()




'''
if local_runtime:
    from IPython.display import Image, display
    import tempfile
    import os.path as osp
    import cv2
    with tempfile.TemporaryDirectory() as tmpdir:
        file_name = osp.join(tmpdir, 'pose_results.png')

        cv2.imshow("demo",vis_result)
        cv2.waitKey(10000)
        # cv2.imwrite(file_name, vis_result[:,:,::-1])
        # display(Image(file_name))
else:
    cv2_imshow(vis_result[:,:,::-1]) #RGB2BGR to fit cv2
'''