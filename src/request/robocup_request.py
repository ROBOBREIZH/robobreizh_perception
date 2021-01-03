from src.request.request import Request

class RobocupRequest(Request):
    def __init__(self, data=None, isWaving=[], isChairEmpty=[], isChairTaken=[], isPerson=[], isBag=[], features=[], img=[]):
        super().__init__()
        self.data = data
        self.isWaving = isWaving
        self.isChairEmpty = isChairEmpty
        self.isChairTaken = isChairTaken
        self.isPerson = isPerson
        self.isBag = isBag
        self.features = features
        self.img = img


