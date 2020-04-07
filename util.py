from torch.autograd.variable import Variable
import torch


class TensorGenerator:
    def get_target_true(size):
        '''
        Tensor containing ones, with shape = size
        '''
        data = Variable(torch.ones(size, 1))
        if torch.cuda.is_available(): return data.cuda()
        return data

    def get_target_false(size):
        '''
        Tensor containing zeros, with shape = size
        '''
        data = Variable(torch.zeros(size, 1))
        if torch.cuda.is_available(): return data.cuda()
        return data

    def get_random_vector(size):
        '''
        Generates a 1-d vector of gaussian sampled random values
        '''
        n = Variable(torch.randn(size, 100))
        if torch.cuda.is_available(): return n.cuda()
        return n


class ImageVectors:
    def get_vector(images):
        return images.view(images.size(0), 784)

    def get_image(vectors):
        return vectors.view(vectors.size(0), 1, 28, 28)

