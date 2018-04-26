from weakref import WeakKeyDictionary


class DataCache:

    def __init__(self):
        self.__data = WeakKeyDictionary()

    def getData(self, key):
        return self.__data.setdefault(key, {})

    def clearData(self, key):
        if key in self.__data:
            del self.__data[key]


_higra_global_cache = DataCache()
