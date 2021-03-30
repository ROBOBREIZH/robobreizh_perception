import os
import sys
import argparse
import torch

sys.path.append(os.getcwd())

from socket import *
import json
from detect import DetectRobocup
from utils.img_conversion import base64_to_cv2

from json import JSONEncoder
from struct import unpack, pack
import numpy


class ServerProtocol:

	def __init__(self):
		self.socket = None
		self.output_dir = '.'
		self.file_num = 1
		self.detector = DetectRobocup()


	def listen(self, server_ip, server_port):
		self.socket = socket(AF_INET, SOCK_STREAM)
		self.socket.bind((server_ip, server_port))
		self.socket.listen(1)

	def handle_images(self):

		try:
			while True:
				(connection, addr) = self.socket.accept()
				try:
					bs = connection.recv(8)
					(length,) = unpack('>Q', bs)
					data = b''
					while len(data) < length:
						# doing it in batches is generally better than trying
						# to do it all in one go
						to_read = length - len(data)
						data += connection.recv(
							4096 if to_read > 4096 else to_read)

					response = json.loads(data.decode('utf-8'))
					img = base64_to_cv2(response['image'])
					objects = response['objects']
					saveImage = response['saveImage']

					result = {}

					if saveImage:
						if "empty_chairs" in objects:
							result["empty_chairs"], result["image_chairs"] = (self.detector.detect_empyt_chairs(img, saveImage))
							objects.remove("empty_chairs")
						if "taken_chairs" in objects:
							result["taken_chairs"], result["image_chairs"] = (self.detector.detect_taken_chairs(img, saveImage))
							objects.remove("taken_chairs")
						if "all_chairs" in objects:
							result["taken_chairs"], result["empty_chairs"], result["image_chairs"] = (self.detector.detect_chairs(img, saveImage))
							objects.remove("all_chairs")
						if "waving_hand" in objects:
							result["waving_hand"], result["image_waving"] = (self.detector.detect_waving_hands(img, saveImage))
							objects.remove("waving_hand")
						if objects:
							r, result["image_object"] = self.detector.detect_objects(img, objects, saveImage)
							result.update(r)
					else:
						if "empty_chairs" in objects:
							result["empty_chairs"] = (self.detector.detect_empyt_chairs(img))
							objects.remove("empty_chairs")
						if "taken_chairs" in objects:
							result["taken_chairs"] = (self.detector.detect_taken_chairs(img))
							objects.remove("taken_chairs")
						if "all_chairs" in objects:
							result["taken_chairs"], result["empty_chairs"] = (self.detector.detect_chairs(img))
							objects.remove("all_chairs")
						if "waving_hand" in objects:
							result["waving_hand"] = (self.detector.detect_waving_hands(img))
							objects.remove("waving_hand")
						if objects:
							r = self.detector.detect_objects(img, objects)
							result.update(r)

					class NumpyArrayEncoder(JSONEncoder):
					    def default(self, obj):
					        if isinstance(obj, numpy.ndarray):
					            return obj.tolist()
					        return JSONEncoder.default(self, obj)

					res_bytes = json.dumps(result, cls=NumpyArrayEncoder).encode('utf-8') 
					print("Image analyzed, sending back masks")

					length = pack('>Q', len(res_bytes))

					connection.sendall(length)

					connection.sendall(res_bytes)
					print("Data sent")

				finally:
					connection.shutdown(SHUT_WR)
					connection.close()

				self.file_num += 1
		finally:
			self.close()

	def close(self):
		self.socket.close()
		self.socket = None

if __name__ == '__main__':
	sp = ServerProtocol()
	sp.listen('127.0.0.1', 55555)
	print("Listening")
	sp.handle_images()