import base64

class Request:
    def __init__(self, idPhoto=None):
        self.idPhoto = idPhoto

    def encode64(self, data):
        encoded = base64.b64encode(data)
        return encoded

    def decode64(self, data):
        decoded = base64.b64decode(data)
        return decoded

    def readfile(self, path):
        with open(path, "rb") as file:
            return base64.b64encode(file.read()).decode('utf-8')

    def get_data(self):
        return self.__dict__
