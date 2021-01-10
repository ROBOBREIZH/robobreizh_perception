import json
import sys
import time
import traceback

import tornado.escape
import tornado.gen
import tornado.web

from src.request.robocup_request import RobocupRequest
from src.detection.detect import Detect
from src.utils.img_conversion import base64_to_cv2, cv2_to_base64, base64_to_video, cv2_to_base64_png

class RobocupHandler(tornado.web.RequestHandler):

    def set_default_headers(self, *args, **kwargs):
        self.set_header("Content-Type", "application/json")
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header("Access-Control-Allow-Methods", "*")

    def _return_response(self, request, message_to_be_returned, status_code):
        """
        Returns formatted response back to client
        """
        try:
            request.set_header("Content-Type", "application/json; charset=UTF-8")
            request.set_status(status_code)

            #If dictionary is not empty then write the dictionary directly into
            if(bool(message_to_be_returned)):
                request.write(message_to_be_returned)

            request.finish()
        except Exception:
            raise

    def prepare(self):
        print("Message received.")

    def data_received(self, chunk):
        print("Data received.", chunk)

    def get(self):
        print("GET: Received!")

    def post(self):
        """
        This function parses the request body, analyse the image and send back the results.
        """
        try:
            t = time.time()
            request_payload = tornado.escape.json_decode(self.request.body)
            input_image = request_payload["image"]

            objNames = ["displayBag", "displayChair", "displayPerson", "displayPose", "displayWaving", "isMovement", "onlyMovement", "features", "isVideo"]
            disp = []
            for d in objNames:
                if d in request_payload:
                    disp.append(request_payload[d])
                else:
                    disp.append(True)
            detect = Detect(displayBag=disp[0], displayChair=disp[1], displayPerson=disp[2], displayPose=disp[3], displayWaving=disp[4], isMovement=disp[5], onlyMovement=disp[6], features=disp[7], isVideo=disp[8])

            output_image, isWaving, isBag, isPerson, isChairEmpty, isChairTaken, features = detect.detect_image(base64_to_cv2(input_image))
            data = cv2_to_base64(output_image).decode('utf-8')

            # read the image created from pepper camera
            #img = cv2.imread("./data/demo/demo_test.png")
            # encode the image in order to send it tp the other pc
            #img_encoded = cv2_to_base64_png(img).decode("utf-8")

            output = RobocupRequest(
                #idPhoto=request_payload["idPhoto"],
                data = data,
                isWaving=isWaving,
                isBag=isBag,
                isPerson=isPerson,
                isChairEmpty=isChairEmpty,
                isChairTaken=isChairTaken,
                features=features
                #img=img_encoded
            )

            print('Total time. (%.3fs)' % (time.time() - t))
            return self._return_response(self, message_to_be_returned=tornado.escape.json_encode(output.get_data()), status_code=200)

        except json.decoder.JSONDecodeError:
            return self._return_response(self, { "message": 'Cannot decode request body!' }, 400)

        except Exception as ex:
            return self._return_response(self, { "message": 'Could not complete the request because of some error at the server!', "cause": ex.args[0], "stack_trace": traceback.format_exc(sys.exc_info()) }, 500)
