import threading
import pickle

class ImageCache():
    def __init__(self):
        self.lock = threading.Lock()
        self.data = {}

    def __len__(self):
        return len(self.data)

    def get_img(self,img_idx):
        '''
        Get an image from the cache based on key
        :param img_idx: key used to store image
        :return: numpy image array
        '''
        self.lock.acquire()
        img_ary = self.data.get(img_idx)
        self.lock.release()
        return img_ary

    def add_img(self,img_idx,img_ary):
        '''
        Add an image to the cache.
        :param img_idx: name of key to be used for later retreival
        :param img_ary: numpy image array
        :return: void
        '''
        self.lock.acquire()
        self.data[img_idx] = img_ary
        self.lock.release()

    def dump_cache(self,pkl_name):
        '''
        Save the cached values to a pickle file for subsequent experiments.
        :param pkl_name: full path to fie
        :return:
        '''
        self.lock.acquire()
        try:
            file = open(pkl_name, 'wb')
            pickle.dump(self.data, file)
            file.close()
            self.lock.release()
            return True
        except:
            self.lock.release()
            return False

    def preload(self,pkl_name):
        '''
        Warm the cache with a pickle file.
        :param pkl_name: full path to pickle file
        :return: BOOl success
        '''
        try:
            file = open(pkl_name, 'rb')
            self.data = pickle.load(file)
            file.close()
        except:
            return False
        return True