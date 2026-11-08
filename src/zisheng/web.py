"""Serve the trained pilot as a small, local-only interactive web app."""

from __future__ import annotations

import base64
from collections.abc import Callable
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
from pathlib import Path
import time
import webbrowser

import numpy as np
from PIL import Image
import torch

from .data import PILOT_CHARACTERS
from .diffusion import GaussianDiffusion
from .model import ConditionalDenoiser
from .train import choose_device


STREAM_FRAME_INTERVAL = 10


def _png_data_url(image: Image.Image) -> str:
    """Encode a small figure directly into the response; no files are written."""

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _render_sample_images(model_images: torch.Tensor) -> list[str]:
    """Encode each 32×32 model output separately, without making a montage."""

    images = np.asarray(model_images.detach().cpu().numpy(), dtype=np.float32).squeeze(1)
    rendered: list[str] = []
    for model_image in images:
        # Model tensors use +1 for paper and -1 for ink. Keep native resolution;
        # the browser scales each independent sample with nearest-neighbor pixels.
        pixels = np.clip((model_image + 1.0) * 127.5, 0, 255).astype(np.uint8)
        rendered.append(_png_data_url(Image.fromarray(pixels, mode="L")))
    return rendered


class DiffusionRunner:
    """Load weights once and run fresh reverse chains for browser requests."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        requested_device: str,
    ) -> None:
        checkpoint = checkpoint.expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        self.device = choose_device(requested_device)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.config = saved["config"]
        self.characters = self.config.get(
            "characters",
            [
                {
                    "glyph": item.glyph,
                    "class_id": item.class_id,
                    "stroke_count": item.stroke_count,
                    "gloss": item.gloss,
                }
                for item in PILOT_CHARACTERS
            ],
        )
        self.by_glyph = {item["glyph"]: item for item in self.characters}
        self.model = ConditionalDenoiser(num_classes=int(self.config["num_classes"]))
        self.model.load_state_dict(saved["model_state_dict"])
        self.model.to(self.device).eval()
        self.diffusion = GaussianDiffusion(
            timesteps=int(self.config["timesteps"]),
            beta_start=float(self.config["beta_start"]),
            beta_end=float(self.config["beta_end"]),
        ).to(self.device)

    def _prepare_request(
        self, payload: dict[str, object]
    ) -> tuple[str, dict[str, object], int, int, torch.Tensor, int]:
        """Validate controls and prepare the batch once for either API path."""
        glyph = str(payload.get("character", ""))
        if glyph not in self.by_glyph:
            raise ValueError("Choose one of the three characters shown in the page.")
        character = self.by_glyph[glyph]

        count = int(payload.get("count", 4))
        if not 1 <= count <= 6:
            raise ValueError("Sample count must be between 1 and 6.")
        seed = int(payload.get("seed", 43))
        if not 0 <= seed <= 2**31 - 1:
            raise ValueError("Seed must be between 0 and 2,147,483,647.")

        character_ids = torch.full(
            (count,), int(character["class_id"]), dtype=torch.long, device=self.device
        )
        total = int(self.config["timesteps"])
        return glyph, character, count, seed, character_ids, total

    def _frame(
        self,
        remaining_steps: int,
        images: torch.Tensor,
        *,
        total: int,
    ) -> dict[str, object]:
        """Encode the independent sample states at one point in the chain."""
        if remaining_steps == total:
            label = "Starting Gaussian noise · x_T"
        elif remaining_steps == 0:
            label = "Final generated glyph · x_0"
        else:
            label = f"Step {remaining_steps} · {remaining_steps} noise steps remain"
        return {
            "remaining_steps": remaining_steps,
            "label": label,
            "images": _render_sample_images(images),
        }

    def generate(self, payload: dict[str, object]) -> dict[str, object]:
        """Sample a complete batch and return separate images plus five states."""
        glyph, character, count, seed, character_ids, total = self._prepare_request(payload)
        snapshot_steps = sorted(
            {0, total, round(total * 0.25), round(total * 0.5), round(total * 0.75)}
        )
        started = time.perf_counter()
        with torch.inference_mode():
            generated, snapshots = self.diffusion.sample(
                self.model,
                character_ids,
                image_size=int(self.config["image_size"]),
                seed=seed,
                snapshot_steps=snapshot_steps,
            )
        elapsed = time.perf_counter() - started
        snapshots[0] = generated.detach().cpu()

        trajectory: list[dict[str, object]] = []
        for remaining_steps in sorted(snapshots, reverse=True):
            trajectory.append(
                self._frame(
                    remaining_steps,
                    snapshots[remaining_steps],
                    total=total,
                )
            )

        return {
            "character": glyph,
            "gloss": character["gloss"],
            "stroke_count": character["stroke_count"],
            "seed": seed,
            "device": str(self.device),
            "timesteps": total,
            "elapsed_seconds": round(elapsed, 2),
            "sample_count": count,
            "samples": _render_sample_images(generated),
            "trajectory": trajectory,
        }

    def generate_stream(
        self,
        payload: dict[str, object],
        emit: Callable[[dict[str, object]], None],
    ) -> dict[str, object]:
        """Stream actual reverse states while the sampler is running.

        A frame is sent every ten DDPM steps. Rendering and transferring those
        states adds some time, but lets the page show the model as it runs.
        """
        glyph, character, count, seed, character_ids, total = self._prepare_request(payload)
        frame_count = 0

        def send_frame(remaining_steps: int, images: torch.Tensor) -> None:
            nonlocal frame_count
            frame = self._frame(
                remaining_steps,
                images,
                total=total,
            )
            emit({"type": "frame", **frame})
            frame_count += 1

        started = time.perf_counter()
        with torch.inference_mode():
            generated, _ = self.diffusion.sample(
                self.model,
                character_ids,
                image_size=int(self.config["image_size"]),
                seed=seed,
                step_callback=send_frame,
                callback_interval=STREAM_FRAME_INTERVAL,
            )
        elapsed = time.perf_counter() - started
        return {
            "type": "complete",
            "character": glyph,
            "gloss": character["gloss"],
            "stroke_count": character["stroke_count"],
            "seed": seed,
            "device": str(self.device),
            "timesteps": total,
            "elapsed_seconds": round(elapsed, 2),
            "sample_count": count,
            "frame_count": frame_count,
        }


class DemoRequestHandler(BaseHTTPRequestHandler):
    """Serve one HTML page and a small JSON generation endpoint."""

    server_version = "ZiShengLocal/0.1"

    def __init__(self, *args: object, runner: DiffusionRunner, page: bytes, **kwargs: object):
        self.runner = runner
        self.page = page
        super().__init__(*args, **kwargs)

    def _send(self, status: int, content_type: str, content: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, "application/json; charset=utf-8", body)

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        route = self.path.split("?", maxsplit=1)[0]
        if route == "/":
            self._send(200, "text/html; charset=utf-8", self.page)
        elif route == "/api/health":
            self._send_json(
                200,
                {
                    "status": "ready",
                    "device": str(self.runner.device),
                    "timesteps": int(self.runner.config["timesteps"]),
                    "image_size": int(self.runner.config["image_size"]),
                    "characters": self.runner.characters,
                },
            )
        else:
            self._send(404, "text/plain; charset=utf-8", b"Not found")

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path not in {"/api/generate", "/api/stream"}:
            self._send(404, "text/plain; charset=utf-8", b"Not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 8192:
                raise ValueError("Request body is missing or too large.")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object.")
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._send_json(400, {"error": str(error)})
            return

        if self.path == "/api/generate":
            try:
                result = self.runner.generate(payload)
            except (ValueError, TypeError) as error:
                self._send_json(400, {"error": str(error)})
                return
            except Exception as error:  # Report a useful local message without killing the server.
                self._send_json(500, {"error": f"Generation failed: {error}"})
                return
            self._send_json(200, result)
            return

        headers_sent = False

        def emit(event: dict[str, object]) -> None:
            nonlocal headers_sent
            if not headers_sent:
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Cache-Control", "no-cache, no-transform")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                headers_sent = True
            line = json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n"
            self.wfile.write(line)
            self.wfile.flush()

        try:
            completion = self.runner.generate_stream(payload, emit)
            emit(completion)
        except (BrokenPipeError, ConnectionResetError):
            # Stop inference if the browser leaves before the stream completes.
            return
        except (ValueError, TypeError) as error:
            if headers_sent:
                emit({"type": "error", "error": str(error)})
            else:
                self._send_json(400, {"error": str(error)})
        except Exception as error:  # Report useful details in the local page.
            if headers_sent:
                emit({"type": "error", "error": f"Generation failed: {error}"})
            else:
                self._send_json(500, {"error": f"Generation failed: {error}"})

    def log_message(self, format: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}", flush=True)


def serve(
    checkpoint: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    requested_device: str = "auto",
    open_browser: bool = True,
) -> None:
    """Start the single-user website on localhost and optionally open it."""

    page_path = Path(__file__).with_name("web_ui.html")
    if not page_path.exists():
        raise FileNotFoundError(f"Web interface not found: {page_path}")
    page = page_path.read_bytes()
    runner = DiffusionRunner(
        checkpoint,
        requested_device=requested_device,
    )
    handler = partial(DemoRequestHandler, runner=runner, page=page)
    server = HTTPServer((host, port), handler)
    address = f"http://{host}:{port}"
    print(f"字生 is ready at {address}", flush=True)
    print("Keep this terminal open while using the page. Press Ctrl+C to stop it.", flush=True)
    if open_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping the local 字生 server.", flush=True)
    finally:
        server.server_close()
