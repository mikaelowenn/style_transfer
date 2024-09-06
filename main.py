import tensorflow as tf
from src.data_loader import load_and_preprocess_image
from src.model import get_style_model_and_losses
from src.train import fit_style_transfer

def main():
    content_image_path = 'data/content_images/samsung-memory-nplkFSNschY-unsplash.jpg'
    style_image_path = 'data/style_images/madonna-and-child-enthroned.jpg!Large.jpg'
    
    content_image = load_and_preprocess_image(content_image_path)
    style_image = load_and_preprocess_image(style_image_path)
    
    print("Content Image Shape:", content_image.shape)
    print("Style Image Shape:", style_image.shape)

    content_layers = ['block5_conv2'] 
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

    extractor = get_style_model_and_losses(style_layers, content_layers)

    stylized_image = fit_style_transfer(content_image, style_image, extractor)
    
    print("Stylized Image Shape:", stylized_image.shape)

if __name__ == "__main__":
    main()