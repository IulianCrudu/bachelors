import typing as t

from yolo_utils import *
from utils import ImageInfo


device = torch.device('cpu')


def load_model():
    model = torch.jit.load("models/yolopv2.pt", map_location='cpu')
    model = model.to(device)
    model.eval()

    return model


def vehicle_tracking(source: str, save_source: str = "runs/detect") -> t.List[ImageInfo]:
    # source = "data/leftImg8bit_trainvaltest/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png"
    imgsz = 640
    stride = 32
    name = "exp"
    exist_ok = True
    conf_thres = 0.3
    iou_thres = 0.45
    agnostic_nms = True
    save_txt = True
    save_conf = False
    save_img = True

    coordinates = []
    model = load_model()
    save_dir = Path(increment_path(Path(save_source) / name, exist_ok=exist_ok))  # increment run

    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    for path, img, im0s, vid_cap in dataset:
        current_coordinates = []
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        [pred, anchor_grid], seg, ll = model(img)

        # waste time: the incompatibility of  torch.jit.trace causes extra time consumption in demo version
        # but this problem will not appear in offical version
        pred = split_for_trace_model(pred, anchor_grid)

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=agnostic_nms)

        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xyxy_tensor = torch.tensor(xyxy).view(1, 4)
                    current_coordinates.append(xyxy_tensor[0])
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(xyxy_tensor) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *(xyxy_tensor.view(-1).tolist()), conf) if save_conf else (cls, *(xyxy_tensor.view(-1).tolist()))  # label format
                        # line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img:  # Add bbox to image
                        plot_one_box(xyxy, im0, line_thickness=3)

            # Print time (inference)
            show_seg_result(im0, (da_seg_mask, ll_seg_mask), is_demo=True)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            # w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            # h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            w, h = im0.shape[1], im0.shape[0]
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
        coordinates.append(ImageInfo(path, current_coordinates))

    return coordinates


if __name__ == "__main__":
    source = "data/data_tracking_image_2 KITTI/testing/image_02/0000/000000.png"
    vehicle_tracking(source=source)
