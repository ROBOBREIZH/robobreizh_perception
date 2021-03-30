import os
import sys
import argparse
import base64
import time
from struct import pack, unpack

sys.path.append(os.getcwd())

from socket import *
import json

def image_request(sock, img_file, obj):
	print("Request detection . . . \n")
	#print("Connection on {}".format(port))

	with open(img_file, mode='rb') as file:
		img = file.read()
	
	encoding = {'image': base64.encodebytes(img).decode('utf-8'), 'objects': obj}

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


if __name__ == '__main__':

	hote = "127.0.0.1"
	port = 55555

	sock = socket(AF_INET, SOCK_STREAM)
	sock.connect((hote, port))

	img_file = "test/table.jpg"
	obj = ["all"]

	image_request(sock, img_file, obj)
	sock.close()
	
	time.sleep(2)

	sock = socket(AF_INET, SOCK_STREAM)
	sock.connect((hote, port))

	img_file = "test/waving-hand.jpg"
	obj = ["waving_hand"]

	image_request(sock, img_file, obj)

	sock.close()