import configparser
import torch

class NewsAnalyzer:
    # annoate patterns
    def __init__(self):
        
        # load the configuration file "config.ini"
        config = configparser.ConfigParser()
        config.read("config.ini")
        
        # decide in which environment you are in
        try:
            # first try google.colab
            from google.colab import drive
            self.in_colab = True
            print("in Google COLAB: load Google Drive and use GPU")
            self.hostname = "GOOGLE_COLAB"
            #set the data folder path
            drive.mount('/content/drive')
        except:
            # we are not in google.colab
            print("not in Google COLAB: access the data folder and use CPU")
            self.in_colab = False
            # check what the hostname is
            import socket
            self.hostname=socket.gethostname()
            print("hostname = {}".format(self.hostname))
            #set the data folder path
            DATA_DIR = "/media/data/kfunaya/mvp21/samples/"
            
        # set the path parameters
        self.data_dir = config[self.hostname]['data_dir']
        self.model_dir = config[self.hostname]['model_dir']
        self.label_path = config[self.hostname]['label_path']
        
        if self.in_colab:
            # load CUDA
            assert(torch.cuda.is_available())
            self.has_cuda = True
            self.device = torch.device("cuda")
        else:
            try:
                # check if CUDA device exists
                assert(torch.cuda.is_available())
                self.has_cuda = True
            except:
                # CUDA device does not exist
                self.has_cuda = False
            # in either case, we don't load CUDA, just use CPU.
            self.device = torch.device("cpu")
            
        pass


