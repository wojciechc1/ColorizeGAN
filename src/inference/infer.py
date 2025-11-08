from src.inference.colorizer import Colorizer
import matplotlib.pyplot as plt
from src.config import CONFIG


def main():
    input_image_path = "../../docs/output1.png"
    # output_image_path = "output8.png"

    colorizer = Colorizer(CONFIG["generator_path"], CONFIG["device"])

    image = colorizer(input_image_path)

    plt.imshow(image)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()


#TODO add CI / update README / update GUI