class Context(object):
    @property
    def context_data(self):
        raise NotImplementedError("Abstract method")