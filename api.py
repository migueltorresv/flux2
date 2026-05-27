import io
import os
import random
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from einops import rearrange
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent / "src"))

from flux2.sampling import (
    batched_prc_img,
    batched_prc_txt,
    denoise,
    encode_image_refs,
    get_schedule,
    scatter_ids,
)
from flux2.util import load_ae, load_flow_model, load_text_encoder

MODEL_NAME = os.getenv("MODEL_NAME", "flux.2-klein-4b")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading model: {MODEL_NAME} on {DEVICE}")
    models["text_encoder"] = load_text_encoder(MODEL_NAME, device=DEVICE)
    models["flow"] = load_flow_model(MODEL_NAME, device=DEVICE)
    models["ae"] = load_ae(MODEL_NAME, device=DEVICE)
    print("Models loaded successfully.")
    yield
    models.clear()


app = FastAPI(title="FLUX.2 Image Generation API", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt to generate the image from")
    width: int = Field(default=1360, ge=64, le=2048, multiple_of=16)
    height: int = Field(default=768, ge=64, le=2048, multiple_of=16)
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


@app.post("/generate", response_class=Response, responses={200: {"content": {"image/png": {}}}})
def generate(req: GenerateRequest):
    text_encoder = models["text_encoder"]
    flow_model = models["flow"]
    ae = models["ae"]

    seed = req.seed if req.seed is not None else random.randrange(2**31)

    try:
        with torch.no_grad():
            ref_tokens, ref_ids = encode_image_refs(ae, [])

            ctx = text_encoder([req.prompt]).to(torch.bfloat16)
            ctx, ctx_ids = batched_prc_txt(ctx)

            shape = (1, 128, req.height // 16, req.width // 16)
            generator = torch.Generator(device=DEVICE).manual_seed(seed)
            randn = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device=DEVICE)
            x, x_ids = batched_prc_img(randn)

            timesteps = get_schedule(4, x.shape[1])
            x = denoise(
                flow_model,
                x,
                x_ids,
                ctx,
                ctx_ids,
                timesteps=timesteps,
                guidance=1.0,
                img_cond_seq=ref_tokens,
                img_cond_seq_ids=ref_ids,
            )
            x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
            x = ae.decode(x).float()

        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")
        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return Response(content=buf.read(), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}
