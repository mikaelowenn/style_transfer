import tensorflow as tf

from src.loss import style_content_loss

@tf.function()
def train_step(image, extractor, optimizer, style_targets, content_targets, content_weight, style_weight):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs, content_targets, style_targets, content_weight, style_weight)

    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))

def fit_style_transfer(content_image, style_image, extractor, num_iterations=10):
    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            # Here you would normally calculate and optimize the loss
            # For this example, we're just passing the image through the extractor
        return image

    image = tf.Variable(content_image)
    
    for i in range(num_iterations):
        image.assign(train_step(image))
        if i % 100 == 0:
            print(f"Iteration {i}")
    
    return image