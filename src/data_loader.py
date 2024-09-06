import tensorflow as tf

def load_and_preprocess_image(image_path, target_size=(256, 256)):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = img / 255.0  # Normalize to [0,1]
    img = tf.expand_dims(img, 0)  # Add batch dimension
    return img

def load_content_style_images(content_path, style_path, target_size=(256, 256)):
    content_image = load_and_preprocess_image(content_path, target_size)
    style_image = load_and_preprocess_image(style_path, target_size)
    print("Content image shape:", content_image.shape)
    print("Style image shape:", style_image.shape)

    return content_image, style_image