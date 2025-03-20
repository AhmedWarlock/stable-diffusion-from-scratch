import numpy as np
import torch
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HIEGHT = 512
LATENT_WIDTH = WIDTH // 8
LATENT_HIEGHT = HIEGHT // 8

def generate( prompt: str, uncond_probpt: str, input_image=None, strength= 0.8,
             do_cfg= True, cfg_scale=7.5, sampler_name="ddpm", num_inference_steps=50, models={}, seed=None,
             device=None, idle_device=None, tokenizer=None
            ):
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # Initialize random number generator according to the seed specified
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)
        
        if do_cfg:
            # Convert into a list of length Seq_Len=77
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            cond_context = clip(cond_tokens)
            # Convert into a list of length Seq_Len=77
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            uncond_context = clip(uncond_tokens)
            # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
            context = torch.cat([cond_context, uncond_context])
        else:
            # Convert into a list of length Seq_Len=77
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            context = clip(tokens)
        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")
        
        latent_shape = (1, 4, LATENT_HIEGHT, LATENT_WIDTH)
        
        if input_image: # Image-to-Image generation
            encoder = models["encoder"]
            encoder.to(device)
            
            input_image_tensor = input_image.resize((HIEGHT, WIDTH))
            input_image_tensor = np.array(input_image_tensor)
            # (HEIGHT, WIDTH, CHANNEL)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (HEIGHT, WIDTH, CHANNEL) --> (BATCH_SIZE, HEIGHT, WIDTH, CHANNEL)
            input_image_tensor = input_image_tensor.unsqueeze(0)
             # ((BATCH_SIZE, HEIGHT, WIDTH, CHANNEL) --> (BATCH_SIZE, CHANNEL, HEIGHT, WIDTH)
            input_image_tensor = input_image_tensor.permute(0, 3, 2, 1)
            
            encoder_noise = torch.randn(latent_shape, generator=generator, device=device)
            # run the image through the VAE encoder
            latents = encoder(input_image_tensor, encoder_noise)
            
            sampler.set_strength(strength = strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])
            
            to_idle(encoder)
            
        else: # Text-to-Image
            latents = torch.randn(latent_shape, generator=generator, device=device)
            
        
        diffusion = models["diffusion"]
        diffusion.to(device) 
        
        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            model_input = latents

            if do_cfg:
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)
            