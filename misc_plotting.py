import torch
import matplotlib.pyplot as plt

def plot_img_for_random_decoded_z(vae, latent_size):

    import matplotlib.pyplot as plt

    # Generate random latent variable z
    z = torch.randn(1, latent_size)

    # Decode z and reconstruct image
    vae.eval()
    with torch.no_grad():
        recon = vae.decode(z)
        recon_image = recon.view(28, 28)

    # Display reconstructed image
    plt.imshow(recon_image.detach().numpy(), cmap='gray')
    plt.axis('off')
    plt.show()

def plot_batch_in_row(images):
    num_images = len(images)
    images = images.view(num_images,28,28).cpu().detach().numpy()
    # Create a figure and axis for plotting
    fig, ax = plt.subplots(1, num_images, figsize=(num_images, 1))
    
    # Iterate through the images and plot them
    for i in range(num_images):
        ax[i].imshow(images[i], cmap='gray')
        ax[i].axis('off')
    
    # Display the row of images
    plt.show()
    
def plot_batches_in_rows(batch1, batch2, path):
    num_images = len(batch1)
    batch1 = batch1.view(num_images,28,28).cpu().detach().numpy()
    batch2 = batch2.view(num_images,28,28).cpu().detach().numpy()
    # Create a figure and axis for plotting
    fig, ax = plt.subplots(2, num_images, figsize=(num_images, 2))
    
    # Iterate through the images in the first batch and plot them
    for i in range(num_images):
        ax[0, i].imshow(batch1[i], cmap='gray')
        ax[0, i].axis('off')
    
    # Iterate through the images in the second batch and plot them
    for i in range(num_images):
        ax[1, i].imshow(batch2[i], cmap='gray')
        ax[1, i].axis('off')
    
    # Display the rows of images
    plt.show()
    plt.savefig(path)