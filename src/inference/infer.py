from src.inference.colorizer import Colorizer
import matplotlib.pyplot as plt
from src.config import CONFIG
from src.utils.visualize import save_colorized_image


def main():
    input_image_path = "../../docs/output1.png"
    output_path = "output8.png"

    colorizer = Colorizer(CONFIG["generator_path"], CONFIG["device"])

    image = colorizer(input_image_path)

    save_colorized_image(image, output_path)
    plt.imshow(image)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()


#TODO add CI / update README / update GUI