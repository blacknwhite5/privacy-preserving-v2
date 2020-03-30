
import torch
class Logger:
    def __init__(self, filename):
        self.filename = filename
        self.f = open(self.filename, 'w')

    def write(self, *args):
        string = ''
        for i in range(len(args)):
            if isinstance(args[i], torch.Tensor):
                string += '{:4f}'.format(args[i])
            else:
                string += str(args[i])


            if i == len(args)-1:
                string += '\n'
            elif isinstance(args[i], str):
                string += ':'
            else:
                string += '\t'
        self.f.write(string)

    def close(self):
        self.f.close()
