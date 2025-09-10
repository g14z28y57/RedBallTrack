import os
import torch
from diffusers import DiffusionPipeline


def create_model():
    # 1. 设置模型和设备
    # SDXL 有两个模型：base (基础模型) 和 refiner (精修模型)
    # 我们将同时加载它们，以获得最佳效果
    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    refiner_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"

    # 检查是否有可用的 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. 加载 SDXL 模型
    # 请注意，这里我们使用 DiffusionPipeline，它能同时处理多个模型
    base = DiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None,
        use_safetensors=True
    )
    base.to(device)

    refiner = DiffusionPipeline.from_pretrained(
        refiner_model_id,
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None,
        use_safetensors=True,
    )
    refiner.to(device)

    return base, refiner


def generate_image(base, refiner, prompt, img_path, img_width, img_height):
    # 4. 生成图像
    # SDXL 的生成过程通常分为两步：
    #   a. 使用基础模型生成初步的图像
    #   b. 使用精修模型进一步提高图像质量
    image = base(
        prompt=prompt,
        num_inference_steps=40,
        denoising_end=0.8,  # 在第80%步时切换到 refiner 模型
        output_type="latent",  # 输出为潜空间表示，方便 refiner 使用
        width=img_width,  # 指定图像宽度
        height=img_height  # 指定图像高度
    ).images

    refined_image = refiner(
        prompt=prompt,
        num_inference_steps=40,
        denoising_start=0.8,  # 从第80%步开始
        image=image
    ).images[0]

    # 5. 保存图像
    refined_image.save(img_path)


def main():
    base, refiner = create_model()
    img_dir = "texture_images"
    prompt = ""
    os.makedirs(img_dir, exist_ok=True)
    for i in range(1683, 2500):
        img_path = os.path.join(img_dir, f"{i}.png")
        generate_image(base, refiner, prompt, img_path, img_width=1024, img_height=1024)


if __name__ == "__main__":
    main()
