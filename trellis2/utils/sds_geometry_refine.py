import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
import os

from diffusers import StableDiffusionImageVariationPipeline
from transformers import CLIPImageProcessor

from .mesh_rasterizer import render_mesh

def get_random_camera(device, dist=2.5, rng=None):
    elev = rng.uniform(-20, 45) if rng else np.random.uniform(-20, 45)
    azim = rng.uniform(0, 360) if rng else np.random.uniform(0, 360)
    
    elev_rad = np.radians(elev)
    azim_rad = np.radians(azim)
    
    x = dist * np.cos(elev_rad) * np.sin(azim_rad)
    y = dist * np.sin(elev_rad)
    z = dist * np.cos(elev_rad) * np.cos(azim_rad)
    
    camera_pos = np.array([x, y, z])
    target = np.array([0, 0, 0])
    up = np.array([0, 1, 0])
    
    z_axis = target - camera_pos
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    extrinsics = np.eye(4)
    extrinsics[:3, 0] = x_axis
    extrinsics[:3, 1] = y_axis
    extrinsics[:3, 2] = z_axis
    extrinsics[:3, 3] = camera_pos

    extrinsics = np.linalg.inv(extrinsics)
    cam_pos_t = torch.tensor(camera_pos, dtype=torch.float32, device=device)
    return torch.tensor(extrinsics, dtype=torch.float32, device=device), cam_pos_t

def refine_geometry(mesh: trimesh.Trimesh, reference_image, n_iters=100, device='cuda', log_every=10, log_dir=None, seed=None) -> trimesh.Trimesh:
    """
    SDS-guided geometry refinement.
    """
    from PIL import Image

    if seed is not None:
        rng = np.random.RandomState(seed)
        torch.manual_seed(seed)
    else:
        rng = np.random.RandomState()

    
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)
    
    offsets = nn.Parameter(torch.zeros_like(vertices))
    optimizer = torch.optim.Adam([offsets], lr=0.003)

    vertex_edge_len = torch.zeros(len(vertices), device=device)
    for i in range(len(vertices)):
        neighbors = mesh.vertex_neighbors[i]
        if len(neighbors) > 0:
            vertex_edge_len[i] = (vertices[i] - vertices[neighbors]).norm(dim=-1).mean()
        else:
            vertex_edge_len[i] = 1e-3
    # Hard per-vertex cap on displacement magnitude. A single noisy SDS step can otherwise
    # spike one vertex independently of its neighbors (observed as vertical "spiking"
    # artifacts) even with the Laplacian/edge regularizers, since those are soft penalties
    # applied after the step, not a hard bound. 0.5x local edge length keeps any single
    # vertex from moving further than half the distance to its neighbors.
    max_offset = (vertex_edge_len * 0.5).unsqueeze(1)

    
    pipe = StableDiffusionImageVariationPipeline.from_pretrained(
        "lambdalabs/sd-image-variations-diffusers", 
        revision="v2.0",
        torch_dtype=torch.float16
    ).to(device)
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.vae = pipe.vae.to(dtype=torch.float32)
    pipe.image_encoder.eval()
    
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    image_input = processor(images=reference_image, return_tensors="pt").pixel_values.to(device).half()
    
    with torch.no_grad():
        image_embeddings = pipe.image_encoder(image_input).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)
        uncond_embeddings = torch.zeros_like(image_embeddings)
        text_embeddings = torch.cat([uncond_embeddings, image_embeddings])
        
    intrinsics = torch.eye(3, device=device)
    intrinsics[0, 0] = 1.2
    intrinsics[1, 1] = 1.2
    intrinsics[0, 2] = 0.5
    intrinsics[1, 2] = 0.5
    
    edge_idx = torch.tensor(mesh.edges, dtype=torch.long, device=device)
    orig_edge_len = (vertices[edge_idx[:, 0]] - vertices[edge_idx[:, 1]]).norm(dim=-1).detach()
    
    # Pre-build Laplacian adjacency
    edges_flat = []
    for i, neighbors in enumerate(mesh.vertex_neighbors):
        for n in neighbors:
            edges_flat.append([i, n])
    lap_edges = torch.tensor(edges_flat, dtype=torch.long, device=device)
    lap_v0 = lap_edges[:, 0]
    lap_v1 = lap_edges[:, 1]
    lap_degrees = torch.zeros(len(vertices), dtype=torch.float32, device=device)
    lap_degrees.scatter_add_(0, lap_v0, torch.ones_like(lap_v0, dtype=torch.float32))
    lap_degrees = lap_degrees.unsqueeze(1).clamp(min=1)
    
    def neighbor_mean_offset(offs):
        neighbor_offs = offs[lap_v1]
        sum_offs = torch.zeros_like(offs)
        sum_offs.scatter_add_(0, lap_v0.unsqueeze(1).expand(-1, 3), neighbor_offs)
        return sum_offs / lap_degrees
    
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
    
    for i in range(n_iters):
        optimizer.zero_grad()
        
        current_vertices = vertices + offsets
        extrinsics, cam_pos = get_random_camera(device, rng=rng)

        out = render_mesh(current_vertices, faces, extrinsics, intrinsics, 256, 256)
        normal = out['normal']
        pos = out['pos']
        mask = out['mask']

        # Diffuse "clay" shading (textureless, geometry-only SDS per this phase's original
        # scope -- see history.md's "MeshAnything V2 integration" section). A raw normal-as-RGB
        # map is wildly
        # out-of-distribution for a natural-image diffusion prior (it's never seen a
        # rainbow tangent-space visualization), which produced incoherent, high-frequency
        # SDS gradients and visible per-vertex spiking. A point light co-located with the
        # camera (standard "headlamp" shading, matching a studio product-photo look) keeps
        # the render photograph-plausible while remaining texture-free.
        light_dir = F.normalize(cam_pos.view(1, 1, 3) - pos, dim=-1)
        shading = (normal * light_dir).sum(-1).clamp(min=0.0)
        ambient, albedo = 0.3, 0.7
        shaded = albedo * (ambient + (1.0 - ambient) * shading)
        render = shaded.unsqueeze(0).expand(3, -1, -1).unsqueeze(0)
        render = render * mask.unsqueeze(0).unsqueeze(0)
        
        if log_dir is not None and i % log_every == 0:
            with torch.no_grad():
                dbg = (render[0].clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
                Image.fromarray(dbg).save(os.path.join(log_dir, f"iter_{i:04d}.png"))
        
        render = F.interpolate(render, size=(512, 512), mode='bilinear', align_corners=False)
        
        render_vae = render * 2.0 - 1.0
        render_vae = render_vae.float()
        
        latents = pipe.vae.encode(render_vae).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor
        latents = latents.half()
        
        noise = torch.randn_like(latents)
        t = torch.randint(200, 800, (1,), device=device).long()
        noisy_latents = pipe.scheduler.add_noise(latents, noise, t)
        
        with torch.no_grad():
            latent_model_input = torch.cat([noisy_latents] * 2)
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            # Lower CFG scale (10 -> 7.5): high guidance disproportionately amplifies the
            # per-pixel variance of a single-sample SDS estimate, which was showing up as
            # incoherent high-frequency noise in the geometry.
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_cond - noise_pred_uncond)

        w = (1 - pipe.scheduler.alphas_cumprod[t]).to(latents.dtype)
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        # Single-sample SDS gradients are known to be noisy at the pixel/latent level
        # (compounded here by VAE-encoder Jacobian artifacts, which characteristically
        # produce grid/stripe patterns). A small spatial blur suppresses the high-frequency
        # component while preserving the low-frequency shape-correcting signal that
        # actually needs to reach the vertices.
        grad = F.avg_pool2d(grad, kernel_size=5, stride=1, padding=2)

        latents.backward(gradient=grad, retain_graph=True)
        
        v0 = current_vertices[edge_idx[:, 0]]
        v1 = current_vertices[edge_idx[:, 1]]
        edge_len = torch.sqrt((v0 - v1).pow(2).sum(-1) + 1e-8)
        edge_reg = ((edge_len - orig_edge_len) ** 2).mean()
        
        lap = offsets - neighbor_mean_offset(offsets)
        laplacian_reg = (lap ** 2).sum(-1).mean()
        
        # Scale weights inversely to mean squared edge length so it works on dense meshes
        global_mean_edge2 = (orig_edge_len.mean() ** 2).clamp(min=1e-6)
        
        # Bumped 6000 -> 30000 (both terms): even after the fp32-VAE/CFG/blur/clamp fixes,
        # a skip-sds A/B comparison showed the with-SDS output still visibly worse than
        # skipping SDS entirely (faceted/blotchy vs. clean cylinder) — the residual ripple
        # was previously assumed cosmetic but isn't. Push regularization much harder before
        # concluding the diffusion prior itself needs to change.
        reg_loss = (edge_reg / global_mean_edge2) * 30000.0 + (laplacian_reg / global_mean_edge2) * 30000.0
        reg_loss.backward()
        
        if not torch.isfinite(offsets.grad).all():
            print(f"iter {i}: non-finite gradient detected, skipping this step")
            optimizer.zero_grad()
            continue
            
        optimizer.step()

        with torch.no_grad():
            offset_norm = offsets.norm(dim=-1, keepdim=True)
            mask = offset_norm > max_offset
            if mask.any():
                offsets[mask.squeeze(-1)] = (offsets / (offset_norm + 1e-8) * max_offset)[mask.squeeze(-1)]

            # Explicit geometric smoothing pass, applied directly to the offsets rather than
            # as a loss term. Pushing the Laplacian *loss* weight up 5x (6000->30000) barely
            # changed the result, because Adam is an adaptive optimizer: it normalizes each
            # parameter's step by its own running gradient-magnitude estimate, so uniformly
            # scaling up one component of the combined gradient doesn't proportionally shrink
            # the step the way it would under plain SGD. A direct blend toward the neighbor
            # mean bypasses Adam entirely and reliably removes high-frequency noise every
            # iteration regardless of gradient-magnitude bookkeeping.
            offsets.data = offsets.data + 0.3 * (neighbor_mean_offset(offsets.data) - offsets.data)



    refined_mesh = mesh.copy()
    refined_mesh.vertices = (vertices + offsets).detach().cpu().numpy()
    
    # Fail loudly if extents grow > 1.5x or if NaN appears
    if not np.isfinite(refined_mesh.vertices).all():
        raise RuntimeError("SDS geometry refinement produced non-finite vertices (NaN/Inf)")
    orig_extents = np.max(mesh.bounds[1] - mesh.bounds[0])
    new_extents = np.max(refined_mesh.bounds[1] - refined_mesh.bounds[0])
    if new_extents > 1.5 * orig_extents:
        raise RuntimeError(f"SDS geometry refinement exploded: extents grew from {orig_extents} to {new_extents}")

    return refined_mesh
