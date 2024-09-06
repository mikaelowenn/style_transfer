# Image Style Transfer Project

This project implements an image style transfer algorithm using TensorFlow/PyTorch.

## Setup

1. Clone this repository
2. Create a virtual environment: `python -m venv style_transfer_env`
3. Activate the virtual environment:
   - On Windows: `style_transfer_env\Scripts\activate`
   - On macOS/Linux: `source style_transfer_env/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

## Usage

1. Place your content image in the data/content_images/ directory.
2. Place your style image in the data/style_images/ directory.
3. Run the style transfer:
python main.py --content_image path/to/content_image.jpg --style_image path/to/style_image.jpg --output_image path/to/output_image.jpg
4. Replace the paths with your actual image file names.
5. The stylized image will be saved in the outputs/ directory.

## Project Structure

- `data/`: Contains content and style images
- `models/`: Stores model weights
- `src/`: Source code for the project
- `tests/`: Unit tests
- `outputs/`: Stores output stylized images
- `main.py`: Main script to run the style transfer

## License

MIT License