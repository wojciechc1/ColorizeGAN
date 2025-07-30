
### Generator U-Net Model:

    Wejście (1x32x32, grayscale)
    ↓
    Encoder (downsampling, ReLU, zapamiętujemy feature mapy)
    ↓
    Bottleneck (najgłębsza reprezentacja)
    ↓
    Decoder (upsampling + skip-connections z encoderem)
    ↓
    Wyjście (3x32x32, kolorowy obraz RGB)
    

![example_image](example.png)
![example_image](example2.png)