import os
import time
import threading
import contextlib
from typing import Dict, Optional, Literal, TypedDict, Any

import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth
from urllib.parse import urlparse, urlunparse

from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor

from requests_testadapter import Resp

from mcp.server.fastmcp import FastMCP


# -----------------------------
# Requests adapter for file://
# -----------------------------
class LocalFileAdapter(requests.adapters.HTTPAdapter):
	def build_response_from_file(self, request):
		file_path = request.url[7:]  # strip "file://"
		if not os.path.isfile(file_path):
			raise FileNotFoundError(f"File not found: {file_path}")

		with open(file_path, "rb") as file:
			buff = bytearray(os.path.getsize(file_path))
			file.readinto(buff)
			resp = Resp(buff)
			r = self.build_response(request, resp)
			return r

	def send(
		self,
		request,
		stream=False,
		timeout=None,
		verify=True,
		cert=None,
		proxies=None,
	):
		return self.build_response_from_file(request)


# ---- Configuration ----
MODEL_PATH = os.environ.get("R4B_MODEL_PATH", "./R-4B")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AuthType = Literal["none", "basic", "digest"]


class CameraConfig(TypedDict):
	url: str
	auth_type: AuthType
	username: Optional[str]
	password: Optional[str]
	verify_tls: bool


SECURITY_CAMERAS: dict[str, CameraConfig] = {
	"ExampleCamera": {
		"url": "http://server/directory/picture",
		"auth_type": "digest",   # try "basic" first; many cameras need "digest"
		"username": "admin",
		"password": "admin",
		"verify_tls": True,
	},
	"BlackbirdNest": {
		"url": "http://192.168.1.1/snap.jpeg",
		"auth_type": "none",   # try "basic" first; many cameras need "digest"
		"username": None,
		"password": None,
		"verify_tls": True,
	},
}

MAX_PROMPT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", "2000"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "2048"))

# ---- Global model state ----
model = None
processor = None
gpu_lock = threading.Lock()


def load_model():
	global model, processor
	if model is not None and processor is not None:
		return

	m = AutoModel.from_pretrained(
		MODEL_PATH,
		torch_dtype=torch.float32,
		local_files_only=True,
		trust_remote_code=True,
	).to(DEVICE)

	p = AutoProcessor.from_pretrained(
		MODEL_PATH,
		local_files_only=True,
		trust_remote_code=True,
	)

	model = m
	processor = p


def _strip_userinfo(url: str) -> str:
	"""Remove user:pass@ from url if present, to avoid leaking creds and parsing issues."""
	p = urlparse(url)
	if p.username or p.password:
		netloc = p.hostname or ""
		if p.port:
			netloc += f":{p.port}"
		return urlunparse((p.scheme, netloc, p.path, p.params, p.query, p.fragment))
	return url


def fetch_image_from_camera(cam: CameraConfig) -> Image.Image:
	session = requests.session()
	session.mount("file://", LocalFileAdapter())

	url = cam["url"]

	try:
		if url.startswith("file://"):
			resp = session.get(url, stream=True)
		else:
			safe_url = _strip_userinfo(url)
			auth = None
			if cam.get("auth_type") == "basic":
				auth = HTTPBasicAuth(cam.get("username") or "", cam.get("password") or "")
			elif cam.get("auth_type") == "digest":
				auth = HTTPDigestAuth(cam.get("username") or "", cam.get("password") or "")

			headers = {
				"User-Agent": "camera-vision-mcp/1.0",
				"Accept": "image/*,*/*;q=0.8",
			}

			resp = session.get(
				safe_url,
				stream=True,
				timeout=10,
				auth=auth,
				headers=headers,
				verify=cam.get("verify_tls", True),
				allow_redirects=True,
			)

			resp.raise_for_status()

			ctype = (resp.headers.get("Content-Type") or "").lower()
			if "image" not in ctype:
				raise ValueError(
					f"Camera did not return an image. Content-Type={ctype} Status={resp.status_code}"
				)

		return Image.open(resp.raw).convert("RGB")

	except requests.HTTPError as e:
		www = ""
		try:
			www = resp.headers.get("WWW-Authenticate", "")
		except Exception:
			pass
		raise PermissionError(f"HTTP error fetching camera image: {e}. WWW-Authenticate={www}") from e
	except Exception as e:
		raise RuntimeError(f"Failed to fetch/parse camera image: {e}") from e


def image_describe(camera: CameraConfig, prompt: str) -> str:
	if processor is None or model is None:
		raise RuntimeError("Model not loaded")

	messages = [
		{
			"role": "user",
			"content": [
				{"type": "image"},
				{"type": "text", "text": prompt},
			],
		}
	]

	text = processor.apply_chat_template(
		messages,
		tokenize=False,
		add_generation_prompt=True,
		thinking_mode="auto",
	)

	image = fetch_image_from_camera(camera)

	inputs = processor(
		images=image,
		text=text,
		return_tensors="pt",
	).to(DEVICE)

	with gpu_lock:
		generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

	output_ids = generated_ids[0][len(inputs.input_ids[0]) :]
	output_text = processor.decode(
		output_ids,
		skip_special_tokens=True,
		clean_up_tokenization_spaces=False,
	)

	# Keep your post-processing
	return output_text.split("</think>\n\n")[-1]


# -----------------------------
# MCP server setup
# -----------------------------
@contextlib.asynccontextmanager
async def app_lifespan(_app: FastMCP):
	# Load once at startup
	load_model()
	yield
	# (optional) cleanup here


mcp = FastMCP(
	name="Camera Vision Service",
	json_response=True,
	lifespan=app_lifespan,
)


# Resources = “GET-like” data access
@mcp.resource("cameras://list")
def list_cameras() -> list[str]:
	"""List available camera names."""
	return sorted(SECURITY_CAMERAS.keys())


@mcp.resource("service://health")
def health() -> Dict[str, Any]:
	"""Service health and configuration."""
	return {
		"status": "ok",
		"device": DEVICE,
		"model_path": MODEL_PATH,
		"cameras": sorted(SECURITY_CAMERAS.keys()),
		"max_prompt_chars": MAX_PROMPT_CHARS,
		"max_new_tokens": MAX_NEW_TOKENS,
	}


# Tools = compute / side-effectful operations
@mcp.tool()
def analyze_camera(camera_name: str, prompt: str) -> Dict[str, Any]:
	"""
	Capture one still image from a named camera and run the vision model on it.

	Args:
		camera_name: One of the configured camera names (ExampleCamera,BlackbirdNest).
		prompt: Instruction/question for the vision model.

	Returns:
		Dict with model output and latency.
	"""
	camera_name = (camera_name or "").strip()
	prompt = (prompt or "").strip()

	if not camera_name:
		raise ValueError("camera_name is required")
	if camera_name not in SECURITY_CAMERAS:
		raise KeyError(
			f"Unknown camera_name '{camera_name}'. Allowed: {sorted(SECURITY_CAMERAS.keys())}"
		)

	if not prompt:
		raise ValueError("prompt is required")
	if len(prompt) > MAX_PROMPT_CHARS:
		raise ValueError(f"prompt too long (>{MAX_PROMPT_CHARS} chars)")

	t0 = time.time()
	output = image_describe(camera=SECURITY_CAMERAS[camera_name], prompt=prompt)
	latency_ms = int((time.time() - t0) * 1000)

	return {
		"camera_name": camera_name,
		"prompt": prompt,
		"output": output,
		"latency_ms": latency_ms,
	}

if __name__ == "__main__":
	import uvicorn

	# GUARANTEED: load before server starts accepting traffic
	load_model()
    
	# Prefer streamable-http ASGI app (python-sdk commonly exposes this)
	try:
		asgi_app = mcp.streamable_http_app()   # endpoint defaults to /mcp (often with trailing slash behavior)
	except AttributeError:
		# Some variants use http_app() instead
		asgi_app = mcp.http_app()

	uvicorn.run(
		asgi_app,
		host=os.environ.get("MCP_HOST", "127.0.0.1"),
		port=int(os.environ.get("MCP_PORT", "8111")),
		log_level="info",
	)
