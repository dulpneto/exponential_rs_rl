

class UnderflowError(Exception):
    """ Result too large to be represented. """
    def __init__(self, message): # real signature unknown
        self.message = message
        super().__init__(message)
