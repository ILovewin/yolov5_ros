# yolov5_ros
把yolov5部署进ros

## 改动

将detect.py文件改写成所需的节点文件，所以这里只上传了yolodetect_node.py这一改写的节点文件

1. ### 终端命令行参数

   `parser.add_argument('--target', type=str, default= '1', help='car/person/obstacleRing')`

   1对应识别遥控小车，2对应识别人，3对应识别障碍环

2. ### 数据加载

   ```
   if webcam: //加载视频流
       view_img = check_imshow(warn=True)
       if target == '1': //下视摄像头
         dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
         bs = len(dataset)
       else: //前视双目相机
         dataset = LoadMyStreams(source, img_size=imgsz, stride=stride, auto=pt, transforms=None)
   ```

   原本视频流的加载方式是使用opencv库打开摄像头获取视频流，但由于前视双目相机在多个模块中使用，而`cv2.VideoCapture`通常在打开摄像头或视频文件后需要释放才能重新打开摄像头，所以需要修改前视双目相机加载数据的方式为接收ros订阅的图像，如下：

   ```
   class LoadMyStreams:
       def __init__(self, sources='/camera/color/image_raw', img_size=640, stride=32, auto=True, transforms=None):
           self.mode = 'stream'
           self.img_size = img_size
           self.stride = stride
           self.sources = [sources]
           self.imgs = [None] * len(self.sources)
           self.bridge = CvBridge()
           self.auto = auto  
           self.transforms = transforms  # 可选的图像变换
           self.image_received = False  # 添加一个标志，表示是否接收到图像
   
           rospy.Subscriber('/camera/color/image_raw', ROSImage, self.update)
   
       def update(self, msg):
           # Convert ROS Image message to OpenCV image
           try:
               cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
               #cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
               self.imgs = cv_image[np.newaxis, ...]  # 添加一个维度表示批处理
               self.image_received = True  # 标记已经接收到图像
           except CvBridgeError as e:
               print(e)
   
       def __iter__(self):
           self.count = -1
           return self
   
       def __next__(self):
           while not self.image_received:
               # 等待图像数据可用
               rospy.sleep(0.1)
           self.count += 1
           if cv2.waitKey(1) == ord('q'):
               cv2.destroyAllWindows()
               raise StopIteration
           
           #im0 = np.array(self.imgs).copy()
           im0 = self.imgs.copy()
   
           
           if self.transforms:
               im = np.stack([self.transforms(x) for x in im0])  # transforms
           else:
               im = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0] for x in im0])  # resize
               im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
               im = np.ascontiguousarray(im)  # contiguous
   
           return self.sources, im, im0, None, ''
       
       def __len__(self):
           return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years
   ```

   这里需注意`self.imgs`的维度处理

   而下视摄像头目前只有一个模块在使用，不会出现同时占用摄像头资源的问题，所以我只修改了打开摄像头的方式为指定的设备文件： `cap = cv2.VideoCapture("/dev/video6")`

3. ### 发布识别数据

   ```
   detection_pub = rospy.Publisher('detection_results', detectPoint, queue_size=10)
   
   # 创建 MyCustomMessage 消息
               detect_point = detectPoint(
                   timestamp=timestamp,
                   x=x_center,
                   y=y_center,
                   state=state,
                   frame_id=frame_id,
                   width = width,
                   height = height
                   #image_data=image_data  # 将图像数据传递给 image_data 字段
               )
               
               # 发布消息到 ROS topic 中
               detection_pub.publish(detect_point)
   
   ```
   
   这里需自定义ros消息包