#!/usr/bin/env python

import rospy
import actionlib
import numpy as np
import time
from socket import *
import json
from sensor_msgs.msg import Image as ImageMsg
import cv2
import base64
from cv_bridge import CvBridge, CvBridgeError
from struct import unpack, pack
import numpy

class Perception:
	def __init__(self) :
		rospy.init_node('perception', anonymous=False)

		self.bridge = CvBridge()
		self.server_ip = rospy.get_param('~server_ip','127.0.0.1')	 
		self.server_port = rospy.get_param('~server_port', 55555)
		self.mode = rospy.get_param('~mode','')
		print("server ip"+self.server_ip)
	def main(self):
		
		if self.mode == "continue":
			self.continuous_detection()
		elif self.mode == "request":
			rospy.spin()

	def continuous_detection(self):
		img_counter = 0

		while(1):
			t = time.time()

			print("===== DETECTION " + str(img_counter) + " ======")
			sock = socket(AF_INET, SOCK_STREAM)

			sock.connect((self.server_ip, self.server_port))

			img = rospy.wait_for_message("/pepper/camera/front/image_raw", ImageMsg) 
			imageF = self.bridge.imgmsg_to_cv2(img, "bgr8")
			obj = ["all"]
			result = self.image_request(sock, imageF, obj, True)
			image_res = np.array(result["image_object"])
			imag = image_res.astype(np.uint8)

			#cv2.imwrite('detection.png',imag)
			print("Time Objects/Person detection. "+str((time.time() - t)))

			cv2.imshow('image_result',imag)
			if (cv2.waitKey(1) & 0xFF == ord('q')):
				break
			img_counter = img_counter + 1 

			sock.close()


	def image_request(self, sock, img, obj, saveImage):
		print("Request detection . . . \n")
		#Parameters: image, objects list (empty_chairs, taken_chairs, all_chairs, waving_hand, all or any other object), saveImage: True or False
		encoding = {'image': self.cv2_to_base64(img).decode('utf-8'), 'objects': obj, 'saveImage': saveImage}

		res_bytes = json.dumps(encoding).encode('utf-8')

		length = pack('>Q', len(res_bytes))
		# sendall to make sure it blocks if there's back-pressure on the socket
		sock.sendall(length)

		sock.sendall(res_bytes)

		bs = sock.recv(8)
		(length,) = unpack('>Q', bs)
		data = b''
		while len(data) < length:
			# doing it in batches is generally better than trying
			# to do it all in one go, so I believe.
			to_read = length - len(data)
			data += sock.recv(4096 if to_read > 4096 else to_read)

		res_dict = json.loads(data.decode('utf-8'))
		print("Data from detection received, number of objects detected: "+str(len(res_dict)))
		print("Names of detected objects: "+str(res_dict.keys()))
		print("\n")
		return res_dict
	
	def cv2_to_base64(self, arr_cv2):
		result, dst_data = cv2.imencode('.jpg', arr_cv2)
		str_base64 = base64.b64encode(dst_data)
		return str_base64

	def base64_to_cv2(self, encoded_data):
		nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
		img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
		return img

if __name__ == '__main__':
	try:
		perception_node = Perception()
		perception_node.main()
	except rospy.ROSInterruptException:
		pass