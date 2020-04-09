from random import randrange

from torch.autograd.variable import Variable
import torch
import pandas as pd


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


class TextVectors:

    def __init__(self):
        text = [[0, "zero", [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                [1, "one", [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
                [2, "two", [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
                [3, "three", [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],
                [4, "four", [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
                [5, "five", [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
                [6, "six", [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
                [7, "seven", [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
                [8, "eight", [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]],
                [9, "nine", [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]]
        self.df = pd.DataFrame(text, columns=['class', 'caption', 'embedding'])

    def get_text_vectors(self, real_labels):
        list_embeddings = []
        list_wrongEmbeddings = []
        for i in range(0, len(real_labels)):
            lab = real_labels[i].item()
            j = 0
            for i in range(0, 5):
                j = randrange(0, 10)
                if j != lab:
                    break
            list_wrongEmbeddings.append(self.df.iat[j, 2])
            list_embeddings.append(self.df.iat[lab, 2])

        return list_embeddings, list_wrongEmbeddings

