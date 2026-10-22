import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import CLIPTokenizer
import torchvision
from PIL import Image
import os
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import glob
from matplotlib import pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CACHE_DIR = "\.cache\huggingface\hub\models--Manojb--stable-diffusion-2-base"

snapshot_dirs = glob.glob(os.path.join(MODEL_CACHE_DIR, "snapshots", "*"))
if not snapshot_dirs:
    raise FileNotFoundError(
        f"No snapshots found under {MODEL_CACHE_DIR}. "
        "Make sure the model was downloaded correctly."
    )
MODEL_PATH = sorted(snapshot_dirs)[-1]
print(f"Loading model from: {MODEL_PATH}")

pipe = StableDiffusionPipeline.from_pretrained(MODEL_PATH, local_files_only=True,).to(device)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

tokenizer = pipe.tokenizer

classification_model = models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V1)
classification_model = classification_model.to(device)
classification_model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def encode_text_batch(prompts):
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        text_embeddings = pipe.text_encoder(
            text_inputs.input_ids.to(device),
            attention_mask=text_inputs.attention_mask.to(device)
        )[0]
    return text_embeddings


prompts=[
    # 150 — sea lion
    "Photo of sea lion, resting on a rock, in a natural coastal environment",

    # 426 — barometer
    "A vintage analog barometer instrument, single object centered on a plain white background, "
    "circular brass dial with numbered pressure scale and pointer needle, polished wood case, "
    "studio product photograph, soft even lighting, sharp focus, high resolution, photorealistic",

    # 843 — swing
    "A single wooden porch swing bench with vertical slatted backrest and armrests, "
    "hanging from two thick ropes attached at each end, white cushion on the seat, isolated on a pure white background. ",

    # 715 — pickelhaube
    "A single Prussian pickelhaube spiked helmet, museum display, centered on a neutral grey background, "
    "polished black leather dome, brass spike on top, metal trim and chinstrap, "
    "studio product photograph, soft directional lighting, sharp focus, high detail, photorealistic",

    # 952 — fig
    "Photo of a fig fruit, cut open, showing seeds, centered on a clean white background",

    # 507 — combination lock
    "A single metal combination padlock with a rotating numbered dial, centered on a plain white background, "
    "brushed silver body, black numbered wheel, shackle visible at top, "
    "studio product photograph, soft even lighting, sharp focus, macro detail, photorealistic",

    # 590 — hand-held computer (PDA)
    "A single handheld PDA device with a small rectangular LCD screen and stylus, "
    "centered on a plain white background, plastic casing with physical buttons below the screen, "
    "studio product photograph, soft even lighting, sharp focus, high detail, photorealistic",

    # 62 — rock python
    "A single rock python snake coiled on dry earth, full body visible, centered in frame, "
    "patterned brown and tan scales with dark blotches, head raised slightly,"]

negative_prompt = ""


torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

labels = [150, 426, 843, 715, 952, 507, 590, 62]

latents_and_labels = []
latents_and_labels_dict = {}

output_dir = "selected_images"
os.makedirs(output_dir, exist_ok=True)

for idx, prompt in enumerate(prompts):
    label = labels[idx]

    text_embeddings = encode_text_batch([prompt])
    uncond_embeddings = encode_text_batch([negative_prompt])
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    pipe.scheduler.set_timesteps(25)

    while True:
        latents = torch.randn(
            (1, 4, 64, 64),
            device=device,
        )
        latents = latents * pipe.scheduler.init_noise_sigma

        with torch.no_grad():
            for t in pipe.scheduler.timesteps:
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings
                ).sample

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + 7.5 * (
                    noise_pred_text - noise_pred_uncond
                )

                latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

            latents = 1 / 0.18215 * latents
            image = pipe.vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()

            img = Image.fromarray((image[0] * 255).astype('uint8'))
            img_tensor = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = classification_model(img_tensor)
                _, predicted_class = torch.max(output, 1)

        if predicted_class.item() == label:
            # Save the accepted image
            save_path = os.path.join(output_dir, f"class_{label}_idx_{idx}.png")
            img.save(save_path)
            print(f"Saved: {save_path}  (prompt: {prompt})")

            latents_and_labels_dict[label] = latents[0]
            break

latent_save_path = "latents.pth"
torch.save(latents_and_labels_dict, latent_save_path)
