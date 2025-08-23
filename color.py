import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import UNet

# --- ustawienia ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "generator3.pth"  # ścieżka do zapisanych wag
input_image_path = "input.jpg"  # Twój obraz do kolorowania
output_image_path = "output.png"  # gdzie zapisać wynik

# --- 1. Inicjalizacja modelu i ładowanie wag ---
g = UNet().to(device)
g.load_state_dict(torch.load(model_path, map_location=device))
g.eval()

# --- 2. Wczytanie obrazu ---
img = Image.open(input_image_path).convert("RGB")

# przeskalowanie do 32x32 jeśli trenowałeś na CIFAR
transform = transforms.Compose([
    #transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])
img_tensor = transform(img).unsqueeze(0).to(device)  # [1,3,H,W]

# --- 3. Konwersja do grayscale ---
gray_tensor = transforms.functional.rgb_to_grayscale(img_tensor, num_output_channels=1)

# --- 4. Generacja kolorowanego obrazu ---
with torch.no_grad():
    fake_color = g(gray_tensor)  # [1,3,H,W]

# przeskalowanie z [-1,1] do [0,1]
fake_color = (fake_color.squeeze(0).cpu() + 1) / 2

# --- 5. Wyświetlenie i zapis ---
fake_np = fake_color.permute(1,2,0).numpy()
plt.imshow(fake_np)
plt.axis("off")
plt.show()

# zapis do pliku
fake_img = Image.fromarray((fake_np * 255).astype("uint8"))
fake_img.save(output_image_path)

print(f"Pokolorowany obraz zapisany w: {output_image_path}")
