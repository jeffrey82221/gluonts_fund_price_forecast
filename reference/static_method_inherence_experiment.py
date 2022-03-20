class Base:
    def __init__(self):
        self.a = 'a'

    def call_static(self):
        type(self).static_method()

    @staticmethod
    def static_method():
        print('in the static method of Base')


class Child(Base):
    def __init__(self):
        super(Child, self).__init__()

    def call_static(self):
        type(self).static_method()


base = Child()
base.call_static()
