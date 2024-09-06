import unittest
import tensorflow as tf
from src.data_loader import load_and_preprocess_image

class TestDataLoader(unittest.TestCase):
    def test_load_and_preprocess_image(self):
        # You need to have a test image in your data directory for this test
        test_image_path = 'data/content_images/samsung-memory-nplkFSNschY-unsplash.jpg'
        
        # Load and preprocess the image
        processed_image = load_and_preprocess_image(test_image_path)
        
        # Check if the output is a tensor
        self.assertIsInstance(processed_image, tf.Tensor)
        
        # Check if the shape is correct (assuming default size of 256x256)
        self.assertEqual(processed_image.shape, (256, 256, 3))
        
        # Check if the values are normalized between 0 and 1
        self.assertTrue(tf.reduce_min(processed_image) >= 0.0)
        self.assertTrue(tf.reduce_max(processed_image) <= 1.0)

if __name__ == '__main__':
    unittest.main()