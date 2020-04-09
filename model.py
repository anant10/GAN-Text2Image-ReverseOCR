from __future__ import absolute_import
import torch
from torch import nn
import pandas as pd
from torch.optim import Adam
from DC_GAN import DiscriminatorNN
from DC_GAN import GeneratorNN
from logger import Logger
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from util import TensorGenerator, ImageVectors, TextVectors
torch.cuda.set_device(0)
torch.set_default_tensor_type('torch.cuda.FloatTensor')


compose = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((.5,), (.5,))
        ])
out_dir = './dataset'
data = datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
num_batches = len(data_loader)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.00, 0.02)

discriminator = DiscriminatorNN()
discriminator.apply(init_weights)

generator = GeneratorNN()
generator.apply(init_weights)

if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()

d_optimizer = Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
loss = nn.BCELoss()


def train_discriminator(optimizer, real_data, fake_data, caption_embedding, wrong_caption_embeddings):
    # Reset gradients
    optimizer.zero_grad()

    # 1. Train on Real Data, real embedding
    prediction_real = discriminator(real_data, caption_embedding)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, TensorGenerator.get_target_true(real_data.size(0)))
    error_real.backward()

    # 2. Train on Real Data, wrong embedding
    prediction_wrong = discriminator(real_data, wrong_caption_embeddings)
    # Calculate error and backpropagate
    error_wrong = loss(prediction_wrong, TensorGenerator.get_target_false(real_data.size(0)))
    error_wrong.backward()

    # 3. Train on Fake Data, right embedding
    prediction_fake = discriminator(fake_data, caption_embedding)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, TensorGenerator.get_target_false(real_data.size(0)))
    error_fake.backward()

    # Update weights with gradients
    optimizer.step()
    final_error = error_real + ((error_fake + error_wrong) / 2.0)
    return final_error, prediction_real, prediction_fake
    return (0, 0, 0)

def train_generator(optimizer, fake_data, caption_embedding):
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data, caption_embedding)
    # Calculate error and backpropagate
    error = loss(prediction, TensorGenerator.get_target_true(prediction.size(0)))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error


num_test_samples = 100
test_noise = TensorGenerator.get_random_vector(num_test_samples)


# Create logger instance
logger = Logger(model_name='GAN-CLS', data_name='MNIST')

text = [[0,"zero",[1,0,0,0,0,0,0,0,0,0]],
        [1,"one",[0,1,0,0,0,0,0,0,0,0]],
        [2,"two",[0,0,1,0,0,0,0,0,0,0]],
        [3,"three",[0,0,0,1,0,0,0,0,0,0]],
        [4,"four",[0,0,0,0,1,0,0,0,0,0]],
        [5,"five",[0,0,0,0,0,1,0,0,0,0]],
        [6,"six",[0,0,0,0,0,0,1,0,0,0]],
        [7,"seven",[0,0,0,0,0,0,0,1,0,0]],
        [8,"eight",[0,0,0,0,0,0,0,0,1,0]],
        [9,"nine",[0,0,0,0,0,0,0,0,0,1]]]

df = pd.DataFrame(text, columns=['class','caption','embedding'])

# Total number of epochs to train
num_epochs = 200
device = torch.device("cuda")

for epoch in range(num_epochs):
    for n_batch, (real_batch, real_labels) in enumerate(data_loader):
        # 1. Train Discriminator
        text_vect = TextVectors()
        list_embeddings, list_wrongEmbeddings = text_vect.get_text_vectors(real_labels)

        list_embeddings = torch.FloatTensor(list_embeddings)
        list_wrongEmbeddings = torch.FloatTensor(list_wrongEmbeddings)

        real_data = Variable(real_batch)
        real_embeddings = Variable(list_embeddings)
        list_wrongEmbeddings = Variable(list_wrongEmbeddings)

        if torch.cuda.is_available():
            real_data = real_data.cuda()
            real_embeddings = real_embeddings.cuda()
            wrong_embeddings = list_wrongEmbeddings.cuda()

        fake_data = generator(TensorGenerator.get_random_vector(real_data.size(0)), real_embeddings).detach()
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer,
                                                                real_data, fake_data, real_embeddings, wrong_embeddings)

        # 2. Train Generator
        # Generate fake data
        fake_data = generator(TensorGenerator.get_random_vector(real_batch.size(0)), real_embeddings)
        # Train G
        g_error = train_generator(g_optimizer, fake_data, real_embeddings)
        # Log error
        logger.log(d_error, g_error, epoch, n_batch, num_batches)

        # Display Progress
        if (n_batch) % 100 == 0:
            # Display Images
            test_images = generator(test_noise, real_embeddings).data.cpu()
            list_real_embeddings = real_embeddings.tolist()
            string8 = ""
            k = 0
            for i in range(0, len(real_embeddings)):
                j = list_real_embeddings[i]
                myList = [round(x) for x in j]
                ind = myList.index(1)
                clas = df.iat[ind, 1]
                string8 = string8 + "   " + clas
                k = k + 1
                if k == 8:
                    print(string8)
                    k = 0
                    string8 = ""
            print(string8)
            logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches);
            # Display status Logs
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                d_error, g_error, d_pred_real, d_pred_fake
            )
        # Model Checkpoints
        logger.save_models(generator, discriminator, epoch)


if (n_batch) % 100 == 0:
    # Display Images
    test_images = generator(test_noise, real_embeddings).data.cpu()
    print(real_embeddings)
    logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches);
    # Display status Logs
    logger.display_status(
        epoch, num_epochs, n_batch, num_batches,
        d_error, g_error, d_pred_real, d_pred_fake
    )