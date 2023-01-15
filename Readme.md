Code from the internet:

* `optical_flow.py` taken from [OpenCV tutorial](https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html) was extended to read the images from the paths and resize them, also returns the flow vectors.
* `vehicle_tracking.py` and `yolo_utils.py` taken from [YOLOPv2 GitHub Repository](https://github.com/CAIC-AD/YOLOPv2). Extended to save and return all the cars' coordinates by using `ImageInfo` and other data structure from the `utils.py` file which is 100% original.
  * `vehicle_tracking.py` is also cleaned up from the CLI-related code and made to work without CUDA.
