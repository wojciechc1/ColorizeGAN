import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
from model import UNet

# settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "saved/generator1.pth"
input_image_path = "imgs/output1.png"
output_image_path = "output8.png"

# model
g = UNet().to(device)
g.load_state_dict(torch.load(model_path, map_location=device))
g.eval()

# load img
img = Image.open(input_image_path).convert("RGB")

transform = transforms.Compose([
    #transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])
img_tensor = transform(img).unsqueeze(0).to(device)  # [1,3,H,W]

gray_tensor = TF.rgb_to_grayscale(img_tensor, num_output_channels=1)  # [1,1,H,W]

# colorize
with torch.no_grad():
    fake_color = g(gray_tensor)  # [1,3,H,W]

# post-processing
fake_color = (fake_color.squeeze(0).cpu() + 1) / 2  # [3,H,W] -> [0,1]
fake_np = fake_color.permute(1,2,0).numpy()


plt.imshow(fake_np)
plt.axis("off")
plt.show()

fake_img = Image.fromarray((fake_np * 255).astype("uint8"))
fake_img.save(output_image_path)

print(f"Pokolorowany obraz zapisany w: {output_image_path}")
