import argparse

import cv2
import tornado.escape
import tornado.httpclient
import tornado.ioloop
import tornado.web

from src.utils.img_conversion import base64_to_cv2, base64_to_video


class MainHandler(tornado.web.RequestHandler):
    def post(self):
        data = self.get_argument('body', 'No data received')
        self.write(data)

application = tornado.web.Application([
    (r"/", MainHandler),
])


def prepare_request(path_to_img, isVideo):
    input = InputRequest("12345678", None, 0.0, isVideo=isVideo)
    input.image = input.readfile(path_to_img)
    input.displayBag = True
    input.displayChair = True
    input.displayPerson = True
    input.displayWaving = False
    input.displayPose = True
    input.withMovement = True
    input.onlyMovement = False
    input.features = True
    input.isVideo = isVideo
    return input.get_data()


def handle_request(response):
    print("Response", response)
    if response.error:

        print("Error:", response.error)
    else:
        print(response.body)
    tornado.ioloop.IOLoop.instance().stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='data/test_client/test.jpg', help='photo path')
    parser.add_argument('--video', type=str, default='data/test_client/cerv.webm', help='video path')
    parser.add_argument('--mode', type=int, default=0, help='0: photo 1: video')
    opt = parser.parse_args()
    print(opt)
    if opt.mode == 0:
        path=opt.image
    else:
        path=opt.video
    application.listen(8890)
    http_client = tornado.httpclient.HTTPClient(defaults=dict(request_timeout=180))

    body = tornado.escape.json_encode(prepare_request(path, opt.mode == 1))  # Make it into a post request
    http = "http://localhost:9989/robocup"
    response = http_client.fetch(http, method='POST', headers=None, body=body)

    output = tornado.escape.json_decode(response.body)

    display = True
    dest= "./data/tmp/received.avi"
    if opt.mode == 0:
        if display:
            print("isBag: ", len(output["isBag"]))
            print("isChairEmpty: ", len(output["isChairEmpty"]))
            print("isWaving: ", output["isWaving"])
            print("features: ", output["features"])
            #img = base64_to_cv2(output["data"])
            #cv2.imshow('image', img)
            #cv2.waitKey(0)
        pass
    else:
        print("Received video and sent it to: %s" % dest)
        print("isBag: ", output["isBag"])
        print("isChairEmpty: ", output["isChairEmpty"])
        print("isWaving: ", output["isWaving"])
        base64_to_video(dest, output["data"])
        pass

    tornado.ioloop.IOLoop.instance().start()