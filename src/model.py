import tensorflow as tf

def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

def get_style_model_and_losses(style_layers, content_layers):
    style_extractor = vgg_layers(style_layers + content_layers)
    return style_extractor