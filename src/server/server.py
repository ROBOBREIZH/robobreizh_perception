#!/usr/bin/env python3.6

# __init__.py

from src.server.robocuphandler import RobocupHandler
from src.utils.conf import SERVER_PORT
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options
from tornado.web import Application

def main():
    """Create and start the server. Makes it available via localhost:9989"""
   
    app = Application([
        ('/robocup', RobocupHandler)
    ])
    http_server = HTTPServer(app)
    http_server.listen(SERVER_PORT)
    print('Listening on http://localhost:%i' % SERVER_PORT)
    IOLoop.current().start()


if __name__ == '__main__':
    main()